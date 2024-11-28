import os
import time
import random
import torch
import theseus as th
import numpy as np

from torch import nn
from tqdm import tqdm
from torch import Tensor
from theseus.geometry import SO3
from torchvision import datasets, transforms

from ..dist import LieDist
from ..utils import ops

from ..noise import PowerNoiseSchedule
from ..metrics import so3 as lie_metrics
from .model import Model, Head
from ..data.symsol import dataset


class Testbed():
    def __init__(self, a):
        self.a = a
        # self.opt = opt
        
        self.noise_schedule = PowerNoiseSchedule(
            alpha_start=self.a.noise_start, 
            alpha_end=self.a.noise_end,
            timesteps=self.a.timesteps,
            power=self.a.power,
        )
        
        repr_size = lie_metrics.get_repr_size(self.a.repr_type)
        size = self.a.img_res
        self.model = Model(in_dim = repr_size,
                           out_dim = repr_size,
                           image_shape = [1, 3, size, size],
                           resnet_depth = self.a.resnet_depth,
                           mlp_layers = self.a.mlp_layers,
                           fourier_block = self.a.fourier_block,
                           activ_fn = self.a.activ_fn
                           )

    # --- inference ---
    def get_flat_batch_test(self, img, n_slices):
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

    def p_sample_apply(self, mu, rt, t):
        batch_size = mu.shape[0]
        t = np.full([batch_size*self.a.n_slices], t, dtype = np.int32)
        # tp = np.full([batch_size], tp, dtype = np.int32)

        sigma_t = self.noise_schedule.sqrt_alphas[t]
        sigma_L = np.full([batch_size*self.a.n_slices], self.a.noise_start, dtype = np.int32)  
        sigma_t = torch.tensor(sigma_t).unsqueeze(dim = 1)  #size(batch*n_slices, 1)
        sigma_L = torch.tensor(sigma_L).unsqueeze(dim = 1)  #size(batch*n_slices, 1)
        
        rt = lie_metrics.as_lie(rt)  #size(batch*n_slices, 3, 3)
        zt = lie_metrics.as_tan(mu)  #size(batch*n_slices, 3)
        # r0 = ops.lsub(rt, SO3.exp_map(sigma_t * zt))  #size(batch, 3, 3)

        step_size = 0.5 * (sigma_t ** 2) / (sigma_L ** 2)  #size(batch*n_slices, 1)
        noise = LieDist._sample_unit(n=(batch_size*self.a.n_slices,)) #size(batch*n_slices, 3)
        rp = ops.add(rt, SO3.exp_map(step_size * zt + torch.sqrt(2 * step_size) * noise))  #size(batch*n_slices, 3, 3)

        # r0 = lie_metrics.as_repr(r0, self.a.repr_type) #size(batch, 3)
        rp = lie_metrics.as_repr(rp, self.a.repr_type) #size(batch*n_slices, 3)

        return rp
    
    def run_test(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])
        test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.a.batch_size)

        cur_time = np.linspace(self.noise_schedule.timesteps, 0, self.a.steps, endpoint = False) -1
        cur_time = cur_time.astype(np.int32).tolist()
        prev_time = cur_time[1:] + [0]
        time_seq = list(zip(cur_time, prev_time))

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = self.model
        model.load_state_dict(torch.load(".pth"))
        model.eval()

        backbone = model.backbone
        head = model.head
        backbone = backbone.to(device)
        head = head.to(device)

        for batch_idx, data in enumerate(test_loader):

            batch = self.get_flat_batch_test(data["img"], self.a.n_slices)
            img = batch["img"].to(device)
            rt = batch["rt"].to(device)

            features = backbone(img)
            for t, tp in time_seq:
                tt = np.full([self.a.batch_size * self.a.n_slices], t, dtype = np.int32)
                mu = head(features, rt, tt)
                rt = self.p_sample_apply(mu, rt, t) #size(batch_size*n_slices, 3)



    # --- train ---
    def get_flat_batch_train(self, img, rot, n_slices):
        # rot: matrix size(batch, 3, 3)
        # img(16, 3, 224, 224) rt(16*256, 3) t(16*256) zt(16*256, 3) r0(16*256, 3)
        batch = {}

        rts, ts, zts, r0s = [], [], [], []

        batch_size = img.shape[0]
        for slice in range(n_slices):
            t = torch.randint(low=0, high=self.noise_schedule.timesteps, size=(batch_size,)) #size(batch,)
            r0 = lie_metrics.as_lie(rot) #size(batch, 3, 3)
            zt = LieDist._sample_unit(n=(batch_size,)) #size(batch, 3)
            rt = ops.add(r0, SO3.exp_map(torch.tensor(self.noise_schedule.sqrt_alphas[t]).unsqueeze(1) * zt))  #size(batch, 3, 3)
            
            rt = lie_metrics.as_repr(rt, self.a.repr_type) #size(batch, 3)
            zt = lie_metrics.as_repr(zt.unsqueeze(1), self.a.repr_type) #size(batch, 1, 3)
            r0 = lie_metrics.as_repr(r0, self.a.repr_type) #size(batch, 3)

            rts.append(rt)
            ts.append(t)
            zts.append(zt.squeeze(1))
            r0s.append(r0)
        
        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim = 0)
        batch["t"] = torch.cat(ts, dim = 0)
        batch["zt"] = torch.cat(zts, dim = 0)
        batch["r0"] = torch.cat(r0s, dim = 0)
        # print(batch['img'].shape, batch['rt'].shape, batch['t'].shape, batch['zt'].shape, batch['r0'].shape)
        return batch


    def train(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])

        train_dataset = dataset.load_symmetric_solids_dataset(split='train', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.a.batch_size)

        
        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = self.model
        model.train()
        model = model.to(device)
        
        optim = torch.optim.AdamW(model.parameters(), lr=self.a.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.a.lr_decay_rate)
        
        for epoch in range(self.a.epochs):
            for batch_idx, (img, rot) in enumerate(train_loader):
                batch = self.get_flat_batch_train(img, rot, self.a.n_slices)
                img = batch["img"].to(device)
                rt = batch["rt"].to(device)
                t = batch["t"].reshape((-1, 1)).to(device) #size(16*256, 1)

                mu = model(img, rt, t)
                ta = batch["zt"].to(device) if self.a.learn_noise else batch["r0"].to(device)
              
                loss = lie_metrics.distance_fn(ta, mu, self.a.loss_name)
                loss = torch.mean(loss)

                optim.zero_grad()
                loss.backward()
                optim.step()
                if (batch_idx + 1) % 100 == 0:
                    print(f"Batch {batch_idx+1}: Train Loss {loss}")

            lr_scheduler.step()



def main():
    random_image = torch.rand(5, 3, 224, 224)
    rot = torch.randn((5, 3, 3))
    mu = torch.randn((5, 1, 3))
    
    class A():
        def __init__(self):
            self.repr_type = "tan"
            self.noise_start = 1e-8
            self.noise_end = 1.0
            self.timesteps = 100
            self.power = 3.0
            self.img_res = 224
            self.resnet_depth = 34
            self.mlp_layers = 1
            self.fourier_block = True
            self.activ_fn = "leaky_relu"
            self.learn_noise = True
            self.loss_name = "chordal"
            self.batch_size = 16
            self.epochs = 150
            self.init_lr = 1e-2
            self.lr_decay_rate = 0.1
            self.n_slices = 128

    # class OPT():
    #     def __init__(self):
    #         self.steps = 100
    #         self.n_slices = 1

    a = A()
    # opt = OPT()
    test = Testbed(a)
    # test.train()
    # r0, rt = test.p_sample_apply(mu, rot, 99, 98)
    # print(r0, rt)
    test.get_flat_batch_test(random_image, a.n_slices)

    return

if __name__ == "__main__":
    main()