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
import csv
from tqdm import tqdm
import datetime
import os
from torch.amp import autocast, GradScaler
import cv2 as cv

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
        
        # Initialize the EMA-model
        decay = self.a.ema_tau
        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    
    # --- inference ---
    def get_flat_batch_test(self, img, n_slices):
        torch.manual_seed(42)
        batch = {}
        batch_size = img.shape[0]
        rts = []
        for _ in range(n_slices):
            rt = SO3.rand(batch_size)
            rt = lie_metrics.as_repr(rt, self.a.repr_type)
            rts.append(rt)

        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim = 0)  #size(batch*n_slice, 3)
        # print(batch["rt"].shape)
        return batch

    def p_sample_apply(self, mu, rt, t):
        size = mu.shape[0] # batch_size*n_slices
        t = np.full([size], t, dtype = np.int32)
        # tp = np.full([batch_size], tp, dtype = np.int32)

        sigma_t = self.noise_schedule.sqrt_alphas[t]
        sigma_L = np.full([size], (self.a.noise_start) ** 0.5, dtype = np.float32)  
        sigma_t = torch.tensor(sigma_t).unsqueeze(dim = 1)  #size(batch_size*n_slices, 1)
        sigma_L = torch.tensor(sigma_L).unsqueeze(dim = 1)  #size(batch_size*n_slices, 1)

        rt = lie_metrics.as_lie(rt)  #size(batch_size*n_slices, 3, 3)
        zt = lie_metrics.as_tan(mu)  #size(batch_size*n_slices, 3)
        # r0 = ops.lsub(rt, SO3.exp_map(sigma_t * zt))  #size(batch, 3, 3)

        epsilon = 2e-6
        step_size = (epsilon * 0.5 * (sigma_t ** 2) / (sigma_L ** 2)).to(rt.device)  #size(batch_size*n_slices, 1)
        noise = (LieDist._sample_unit(n=(size,))).to(rt.device) #size(batch_size*n_slices, 3)
        # print("Noise: ", list(torch.sqrt(2 * step_size) * noise)[0])
        rp = ops.add(rt, SO3.exp_map(step_size * zt / sigma_t.to(rt.device) + torch.sqrt(2 * step_size) * noise))  #size(batch_size*n_slices, 3, 3)

        # r0 = lie_metrics.as_repr(r0, self.a.repr_type) #size(batch, 3)
        rp = lie_metrics.as_repr(rp, self.a.repr_type) #size(batch_size*n_slices, 3)

        return rp
    
    def showImage(self, img):
        # Convert image color
        img = img + 0.5
        img_bgr = cv.cvtColor(np.transpose(img[0].cpu().numpy(), (1, 2, 0)), 
                                cv.COLOR_RGB2BGR)

        cv.imshow("image", img_bgr)

    def test(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])

        test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=transform)
        test_loader = dataset.getDataLoader(test_dataset, batch_size = self.a.batch_size, shuffle=True, num_workers=10)

        cur_time = np.linspace(self.noise_schedule.timesteps, 0, self.a.steps, endpoint = False) -1
        cur_time = cur_time.astype(np.int32).tolist()
        prev_time = cur_time[1:] + [0]
        time_seq = list(zip(cur_time, prev_time))

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = self.model
        model = torch.load(".log/000/model_250000.pth")
        model.eval()

        backbone = model.backbone
        head = model.head
        backbone = backbone.to(device)
        head = head.to(device)

        with torch.no_grad():
            time_pose = []
            all_final_pose = []
            n_slices = 1
            for batch_idx, (img, rotation, rotations_equivalent) in enumerate(test_loader):
                batch = self.get_flat_batch_test(img, n_slices)
                img = batch["img"].to(device)
                rt = batch["rt"].to(device) #size(batch_size*n_slices, 3)
                # print(rt)

                features = backbone(img)
                # print(features)

                for t, tp in time_seq:
                    # print("Time step ", t)
                    tt = torch.tensor(np.full([self.a.batch_size * n_slices, 1], t, dtype = np.int32)).to(device) 
                    mu = head(features, rt, tt)
                    rt = self.p_sample_apply(-mu, rt, t) #size(batch_size*n_slices, 3)
                    time_pose.append(rt[0])
                    print(f"Time step: {t}, Pose: {rt[0]}")

                # print(time_pose)
                print("Ground Truth: ", lie_metrics.as_tan(rotation[:1]))
                print("Predict", rt[0])

                self.showImage(img)

                # Iterate mini-batch
                rt_idx = 0 # rt_idx is the index of rt, size [batch_size*n_slices, 3]
                for sample_idx in range(len(img)):
                    # get ground-truth rotations of current sample
                    rotations = rotations_equivalent[sample_idx].cpu().numpy()
                    print("length", len(img), rt.shape[0])

                    # get predicted rotations of current sample
                    predict_r = lie_metrics.as_mat(rt[rt_idx].unsqueeze(dim=0)).cpu().numpy()[0]

                    # Find the minimum angle from those equivalent answers
                    min_angle = 1000000
                    min_rotations_idx = -1
                    print(f"Find the minimum angle of totally {len(rotations)} possible solutions.")

                    for rot_idx, rotation in enumerate(rotations):
                        # Compute the relative rotation matrix
                        R_rel = np.dot(predict_r.T, rotation)
                        
                        # Calculate the trace
                        trace_R_rel = np.trace(R_rel)
                        
                        # Compute the angular distance (in radians)
                        angle = np.degrees(np.arccos((trace_R_rel - 1) / 2))
                        
                        if angle < min_angle:
                            min_angle = angle
                            min_rotations_idx = rot_idx

                    # Update index of rt
                    rt_idx += 1

                    print(f"Minimum angle: {min_angle}, \npredict: \n{predict_r}, \nmin-corresponding answer: \n{rotations[min_rotations_idx]}")
                    break


                # Wait for user input
                key = cv.waitKey(0)
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord(' '):
                    print("Next image...")
                    cv.destroyAllWindows()
                




    # --- train ---
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

        rts, ts, zts, r0s, tas = [], [], [], [], []

        # torch.manual_seed(42)

        batch_size = img.shape[0]
        for _ in range(n_slices):

            # Sample random timesteps for size = batch_size
            t = torch.randint(low=0, high=self.noise_schedule.timesteps, size=(batch_size,)) #size(batch,)
            # t = torch.tensor([50] * batch_size)
            # Convert rotations to SO(3) representation, type: SO3, size: (batch, 3, 3)
            r0 = lie_metrics.as_lie(rot) 
            
            # Sample unit Gaussian noise, type: tensor, size: (batch, 3)
            zt = LieDist._sample_unit(n=(batch_size,))

            # alphat = torch.tensor(self.noise_schedule.alphas[t]).unsqueeze(1)
            sqrt_alphas_t = torch.tensor(self.noise_schedule.sqrt_alphas[t]).unsqueeze(1)
            # print(sqrt_alphas_t)

            # ta = lie_metrics.as_tan(r0)
            # ta = -1 / sqrt_alphas_t * zt
            ta = -zt

            # Add noise to rotations, type: SO3, size: (batch, 3, 3)
            rt = ops.add(r0, SO3.exp_map(sqrt_alphas_t * zt))
            # print('zt:', zt,'\nrt', SO3.exp_map(sqrt_alphas_t * zt), '\nr0', r0)

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

            # print(alphat, ta, zt)

        # Concatenate slices and add them to the batch
        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim=0)
        batch["t"] = torch.cat(ts, dim=0)
        batch["zt"] = torch.cat(zts, dim=0)
        batch["r0"] = torch.cat(r0s, dim=0)
        batch["ta"] = torch.cat(tas, dim=0)

        # print(batch['img'].shape, batch['rt'].shape, batch['t'].shape, batch['zt'].shape, batch['r0'].shape)
        
        return batch

    def image_show(self, img):
        # # # Display the first image in the batch
        img_vis = img.numpy()
        img_vis = img_vis + 0.5
        # img = images[0].numpy().astype(np.uint8)  # Convert to NumPy array for visualization
        # img = (img - img.min()) / (img.max() - img.min()) * 255

        # print(rot)
        img_vis = np.transpose(img_vis, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        # img = img[..., ::-1]
        plt.imshow(img_vis)  # Display image
        plt.show()

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
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.a.batch_size, shuffle=True, num_workers=20, pin_memory=True)
        train_loader = dataset.getDataLoader(train_dataset, batch_size = self.a.batch_size, shuffle=False, num_workers=10)
        
        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Prepare the model
        model = self.model.to(device)
        model.train()  # Set the model to training mode

        
        # Define optimizer and learning rate scheduler
        optim = torch.optim.AdamW(model.parameters(), lr=self.a.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.a.lr_decay_rate)
        
        # Training data
        train_data_iter = iter(train_loader)
        img, rot, _ = next(train_data_iter)
        batch_idx, epoch_idx = 0, 0

        # Record data and write to csv file
        record_data = []
        avg_loss = 0
        # Get the current date and time
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        current_time = '000'

        # Create folder
        filename = "training.csv"
        dir_path = os.path.join(".log", current_time)

        file_path = os.path.join(dir_path, filename)
        os.makedirs(dir_path, exist_ok=True)

        # streams = [torch.cuda.Stream() for i in range(num_streams)]
        # data = []

        torch.manual_seed(42)

        # Training steps (update parameters)
        for step in tqdm(range(self.a.train_steps), desc="Train loss"):
            # Prepare the training batch
            batch = self.get_flat_batch_train(img, rot, self.a.n_slices)
            img, rt, t, ta = batch["img"], batch["rt"], batch["t"].reshape((-1, 1)), batch['ta']
            img, rt, t, ta = img.to(device), rt.to(device), t.to(device), ta.to(device)

            # print(rt[:64, :2], ta[:64, 2])
        
            # mu = [0 for i in range(num_streams)]
            # loss = [0 for i in range(num_streams)]

            # img_vis = img[0].cpu()
            # self.image_show(img_vis)
            # print(rot[0])

            # Forward pass: predict score values

            mu = model(img, rt, t)

            # print(mu[:2, :5], ta[:2, :5])
            # print(img[:2, 0, 0, 0], rt[:2], t[:2], mu[:2])

            
            # print(t.shape)

            # Compute loss
            loss = torch.mean(lie_metrics.distance_fn(ta, mu, self.a.loss_name))
            avg_loss += loss.item()

            # Backpropagation and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            # for i in range(num_streams):
            #     with torch.cuda.stream(streams[i]):
            #         mu[i] = model(img_ch[i], rt_ch[i], t_ch[i])

            #         # Compute loss
            #         loss[i] = torch.mean(lie_metrics.distance_fn(ta_ch[i], mu[i], self.a.loss_name))
            #         avg_loss += loss[i].item()

            #         # Backpropagation and optimization
            #         loss[i].backward()
            
            # for stream in streams:
            #     stream.synchronize()


            
            # Update ema-model parameters every s steps, [0, s, 2s, ...]
            if batch_idx % self.a.ema_steps == 0:
                self.ema_model.update_parameters(model)

            # Get the next batch of image and gt-rotation matrix from train_data_iter
            try:
                img, rot, _ = next(train_data_iter)
            except StopIteration:
                # Finish iterating all the training data. Next epoch
                train_data_iter = iter(train_loader)
                img, rot, _ = next(train_data_iter)

                # Reset batch index and increase epoch index
                batch_idx = 0
                epoch_idx += 1

                # Update learning rate
                lr_scheduler.step()
                for param_group in optim.param_groups:
                    param_group['lr'] = max(param_group['lr'], self.a.end_lr)
                
                # Print current learning rate
                current_lr = optim.param_groups[0]['lr']
                print(f"Epoch {epoch_idx + 1}: Updated Learning Rate = {current_lr:.6f}")

            # Log progress every b batches, [b-1, 2b-1, 3b-1, ...]
            if (batch_idx + 1) % 20 == 0:
                tqdm.write(f"Batch {batch_idx+1}: Train Loss {avg_loss / 20}")
                record_data.append([step + 1, epoch_idx, batch_idx + 1, avg_loss])
                avg_loss = 0
            
            if (step + 1) % self.a.ckpt_steps == 0:
                model_path = os.path.join(dir_path, f"model_{step + 1}.pth")
                torch.save(model, model_path)
                print(f"Save check-point model to '{model_path}'")

            # Update batch index and epoch index
            batch_idx += 1
        
        print("Finish training process!")




        # Save the training information to csv file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'epoch', 'batch', 'loss'])
            writer.writerows(record_data)
        
        print(f"Training info has been saved to '{file_path}'")

        # Save model
        torch.save(model, os.path.join(dir_path, "model.pth"))

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