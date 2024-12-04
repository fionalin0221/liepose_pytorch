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

        epsilon = 1e-7
        step_size = (epsilon * 0.5 * (sigma_t ** 2) / (sigma_L ** 2)).to(rt.device)  #size(batch_size*n_slices, 1)
        noise = (LieDist._sample_unit(n=(size,))).to(rt.device) #size(batch_size*n_slices, 3)
        rp = ops.add(rt, SO3.exp_map(step_size * zt / sigma_t.to(rt.device) + torch.sqrt(2 * step_size) * noise))  #size(batch_size*n_slices, 3, 3)

        # r0 = lie_metrics.as_repr(r0, self.a.repr_type) #size(batch, 3)
        rp = lie_metrics.as_repr(rp, self.a.repr_type) #size(batch_size*n_slices, 3)

        return rp
    
    def showImage(self, img):
        # [-0.5, 0.5] -> [0, 1]
        img = img + 0.5

        # Convert image color, RGB->BGR
        img_bgr = cv.cvtColor(np.transpose(img[0].cpu().numpy(), (1, 2, 0)), cv.COLOR_RGB2BGR)
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

        # model = self.model

        # Load weights
        model = torch.load(".log/2024-12-03_16-21-07/model.pth", weights_only=False)
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

                # get features of image via backbone network
                features = backbone(img)

                # Denoised pose
                for t, tp in time_seq:
                    tt = torch.tensor(np.full([self.a.batch_size * n_slices, 1], t, dtype = np.int32)).to(device) 
                    mu = head(features, rt, tt)
                    rt = self.p_sample_apply(mu, rt, t) #size(batch_size*n_slices, 3)
                    time_pose.append(rt[0])
                    print(f"Time step: {t}, Pose: {rt[0]}")

                # Iterate mini-batch
                rt_idx = 0 # rt_idx is the index of rt, size [batch_size*n_slices, 3]
                for sample_idx in range(len(img)):
                    # get ground-truth rotations of current sample
                    rotations = rotations_equivalent[sample_idx].cpu().numpy()

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

                self.showImage(img)

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

            # Target of model, formula(9) from paper in 2024 CVPR (Confronting Ambiguity...)
            ta = -zt
            # ta = -1 / sqrt_alphas_t * zt

            # Add noise to rotations, type: SO3, size: (batch, 3, 3)
            rt = ops.add(r0, SO3.exp_map(sqrt_alphas_t * zt))

            # Convert rotation_0, rotation_t, noise_t into the specified representation
            r0 = lie_metrics.as_repr(r0, self.a.repr_type) # size(batch, 3)
            zt = lie_metrics.as_repr(zt, self.a.repr_type) # size(batch, 3)
            rt = lie_metrics.as_repr(rt, self.a.repr_type) # size(batch, 3)
            ta = lie_metrics.as_repr(ta, self.a.repr_type)

            # Collect the slices
            rts.append(rt)
            ts.append(t)
            zts.append(zt)
            r0s.append(r0)
            tas.append(ta)

        # Concatenate slices and add them to the batch, e.g. [img1-1, img1-2, img1-3, ..., img2-1, img2-2, ...]
        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim=0)
        batch["t"] = torch.cat(ts, dim=0)
        batch["zt"] = torch.cat(zts, dim=0)
        batch["r0"] = torch.cat(r0s, dim=0)
        batch["ta"] = torch.cat(tas, dim=0)

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

        torch.manual_seed(42)

        with tqdm(total=self.a.train_steps, desc="Train loss: ", dynamic_ncols=True) as pbar:
            # Training steps (update parameters)
            for step in range(self.a.train_steps):
                # Prepare the training batch
                batch = self.get_flat_batch_train(img, rot, self.a.n_slices)
                img, rt, t, ta = batch["img"], batch["rt"], batch["t"].reshape((-1, 1)), batch['ta']
                img, rt, t, ta = img.to(device), rt.to(device), t.to(device), ta.to(device)

                # Forward pass: predict score values
                mu = model(img, rt, t)

                # Compute loss
                loss = torch.mean(lie_metrics.distance_fn(ta, mu, self.a.loss_name))
                avg_loss += loss.item()

                # Backpropagation and optimization
                optim.zero_grad()
                loss.backward()
                optim.step()
                
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

                # Update tqdm description and progress
                pbar.set_description(f"Train loss: {loss:.4f}")
                pbar.update(1)
                
                # Log progress every b batches, [b-1, 2b-1, 3b-1, ...]
                if (batch_idx + 1) % 20 == 0:
                    tqdm.write(f"Batch {batch_idx+1}: Train Loss {avg_loss / 20}")
                    record_data.append([step + 1, epoch_idx, batch_idx + 1, avg_loss / 20])
                    avg_loss = 0
                
                # save check-point model
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