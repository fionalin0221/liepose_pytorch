import torch
from theseus.geometry import SO3
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# From our customed package
from ..data.symsol import dataset
from ..dist import LieDist
from ..metrics import so3 as lie_metrics
from ..model import Model
from ..noise import PowerNoiseSchedule
from ..utils import ops
import matplotlib.pyplot as plt
import numpy as np
import torch.jit


class Testbed():
    def __init__(self, config):
        """
        Args:
        config (SimpleNamespace): Configuration object containing hyperparameters and settings.
        """
        self.a = config
        
        # Create a noise schedule object for sampling noise during training
        # TODO: Try different noise scheduler.
        self.noise_schedule = PowerNoiseSchedule(
            alpha_start=self.a.noise_start, 
            alpha_end=self.a.noise_end,
            timesteps=self.a.timesteps,
            power=self.a.power,
        )
        
        # get representation size based on the chosen representation
        repr_size = lie_metrics.get_repr_size(self.a.repr_type)
        size = self.a.img_res

        # Initialize the model
        self.model = Model(in_dim = repr_size,
                           out_dim = repr_size,
                           image_shape = [1, 3, size, size],
                           resnet_depth = self.a.resnet_depth,
                           mlp_layers = self.a.mlp_layers,
                           fourier_block = self.a.fourier_block,
                           activ_fn = self.a.activ_fn
                           )
        
        # TODO: Set decay as parameters
        decay = 0.999
        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    
    def get_flat_batch_train(self, img, rot, n_slices):
        """
        Diffusing each image.

        Args: 
        img (torch.Tensor): Input images of shape (batch_size, channels, height, width)
        rot (torch.Tensor): Rotation matrices (label) of shape (batch_size, 3, 3)
        n_slices (int): Number of noisy samples per image.

        Returns:
        dict: A dictionary containing augmented data for training
            - "img" (torch.Tensor): Images, size (batch_size, channels, height, width), e.g., (16, 3, 224, 224)
            - "rt" (torch.Tensor): Noisy rotations, concatenated across slices, size (batch_size * n_slices, 3), e.g., [2048, 3]
            - "t" (torch.Tensor): Noise schedule timesteps, size (batch_size * n_slices), e.g., (2048)
            - "zt" (torch.Tensor): Noise samples in Lie algebra space, size (batch_size * n_slices, 3), e.g., (2048, 3)
            - "r0" (torch.Tensor): Ground truth rotations in Lie algebra space, size (batch_size * n_slices, 3), e.g., (2048, 3)
        """
        batch = {}

        rts, ts, zts, r0s, tas, ats = [], [], [], [], [], []

        torch.manual_seed(42)
        batch_size = img.shape[0]
        for _ in range(n_slices):
            
            # Sample random timesteps for size = batch_size
            t = torch.randint(low=0, high=self.noise_schedule.timesteps, size=(batch_size,)) #size(batch,)
            # t = torch.tensor([50] * batch_size)
            # Convert rotations to SO(3) representation, type: SO3, size: (batch, 3, 3)
            r0 = lie_metrics.as_lie(rot) 
            
            # Sample unit Gaussian noise, type: tensor, size: (batch, 3)
            zt = LieDist._sample_unit(n=(batch_size,)) 

            alphat = torch.tensor(self.noise_schedule.alphas[t]).unsqueeze(1)
            sqrt_alphas_t = torch.tensor(self.noise_schedule.sqrt_alphas[t]).unsqueeze(1)

            # ta = lie_metrics.as_tan(r0)
            # print(r0, ta)
            # ta = -1 / sqrt_alphas_t * zt
            ta = zt

            # Add noise to rotations, type: SO3, size: (batch, 3, 3)
            rt = ops.add(r0, SO3.exp_map(sqrt_alphas_t * zt))
            
            # Convert rotation_0, rotation_t, noise_t into the specified representation
            r0 = lie_metrics.as_repr(r0, self.a.repr_type) # size(batch, 3)
            zt = lie_metrics.as_repr(zt, self.a.repr_type) # size(batch, 3)
            rt = lie_metrics.as_repr(rt, self.a.repr_type) # size(batch, 3)
            ta = lie_metrics.as_repr(ta, self.a.repr_type)

            # print(r0,'\n', rt,'\n', t)
            # Collect the slices
            rts.append(rt)
            ts.append(t)
            zts.append(zt)
            r0s.append(r0)
            tas.append(ta)
            ats.append(alphat.squeeze(1))
            # print(alphat, ta, zt)
        # Concatenate slices and add them to the batch
        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim=0)
        batch["t"] = torch.cat(ts, dim=0)
        # print(ts)
        # print(batch["t"])
        batch["zt"] = torch.cat(zts, dim=0)
        batch["r0"] = torch.cat(r0s, dim=0)
        batch["ta"] = torch.cat(tas, dim=0)
        batch["at"] = torch.cat(ats, dim=0)
        # print(batch["at"].shape)
        # print(batch['img'].shape, batch['rt'].shape, batch['t'].shape, batch['zt'].shape, batch['r0'].shape)
        
        return batch

    def train(self):
        """
        Main training loop for the model
        """
        # Define data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])

        # Load datasets for training and testing
        train_dataset = dataset.load_symmetric_solids_dataset(split='train', transform=transform)
        # test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.a.batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.a.batch_size)
        
        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Prepare the model
        model = self.model
        model.train()  # Set the model to training mode
        model = model.to(device)
        
        # Define optimizer and learning rate scheduler
        optim = torch.optim.AdamW(model.parameters(), lr=self.a.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.a.lr_decay_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(self.a.epochs):
            movavg_loss = 0
            for batch_idx, (img, rot) in enumerate(train_loader):
                # Prepare the training batch
                batch = self.get_flat_batch_train(img, rot, self.a.n_slices)
                img = batch["img"].to(device)
                rt = batch["rt"].to(device)
                t = batch["t"].reshape((-1, 1)).to(device)
                at = batch["at"].to(device)
                r0 = batch["r0"].to(device)

                # # # Display the first image in the batch
                # img_vis = img[0].cpu().numpy()
                # img_vis = img_vis + 0.5
                # # img = images[0].numpy().astype(np.uint8)  # Convert to NumPy array for visualization
                # # img = (img - img.min()) / (img.max() - img.min()) * 255
                # print(rot[0])
                # # print(rot)
                # img_vis = np.transpose(img_vis, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                # # img = img[..., ::-1]
                # plt.imshow(img_vis)  # Display image
                # plt.show()

                # Forward pass: predict score values
                mu = model(img, rt, t)

                # Set target
                ta = batch["ta"].to(device)
 
                


                # Compute loss
                loss = lie_metrics.distance_fn(ta, mu, self.a.loss_name)
                # loss = criterion(ta, mu)
                loss = torch.mean(loss)

                # loss = torch.sum(loss)


                # Backpropagation and optimization
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                # TODO: Set 5 as parameters
                if (batch_idx + 1) % 5 == 0:
                    self.ema_model.update_parameters(model)

                movavg_loss += loss.item()

                # # Inspect gradients
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient for {name}:")
                #         print(param.grad)
                #     else:
                #         print(f"No gradient for {name}")

                # Log progress every xxx batches
                if (batch_idx + 1) % 20 == 0:
                    print(mu[0], ta[0], rt[0], t[0], r0[0])
                    print(f"Batch {batch_idx+1}: Train Loss {movavg_loss / 20}")
                    movavg_loss = 0
            
            # Update learning rate
            lr_scheduler.step()
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{self.a.epochs}: Updated Learning Rate = {current_lr:.6f}")




# def main():
#     random_image = torch.rand(5, 3, 224, 224)
#     rot = torch.randn((5, 3, 3))
    
#     class A():
#         def __init__(self):
#             self.repr_type = "tan"
#             self.noise_start = 1e-8
#             self.noise_end = 1.0
#             self.timesteps = 100
#             self.power = 3.0
#             self.img_res = 224
#             self.resnet_depth = 34
#             self.mlp_layers = 1
#             self.fourier_block = True
#             self.activ_fn = "leaky_relu"
#             self.learn_noise = True
#             self.loss_name = "chordal"
#             self.batch_size = 128
#             self.epoch = 150

#     a = A()
#     test = Testbed(a)
#     test.get_flat_batch_train(random_image, rot, 10)

# if __name__ == "__main__":
#     main()