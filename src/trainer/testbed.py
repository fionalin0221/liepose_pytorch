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
from model import Model
from ..data.symsol import dataset


class Testbed():
    def __init__(self, a):
        self.a = a
        
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
                           img_shape = [1, size, size, 3],
                           resnet_depth = self.a.resnet_depth,
                           mlp_layers = self.a.mlp_layers,
                           fourier_block = self.a.fourier_block,
                           activ_fn = self.a.activ_fn
                           )
        



    
    def get_flat_batch_train(self, img, rot, n_slices):
        # img(16, 1, 224, 224, 3) rt(16, 256, 3) t(16, 256) zt(16, 256, 3) r0(16, 256, 3)
        batch = {}

        rts, ts, zts, r0s = [], [], [], []

        batch_size = img.shape[0]
        for slice in range(n_slices):
            t = torch.randint(low=0, high=self.noise_schedule.timesteps, size=(batch_size,)) #size(batch,)
            r0 = lie_metrics.as_lie(rot) #size(batch, 3, 3)
            zt = LieDist._sample_unit(n=(batch_size,)) #size(batch, 3)
            rt = ops.add(r0, SO3.exp_map(torch.tensor(self.noise_schedule.sqrt_alphas[t]).unsqueeze(1) * zt))  #size(batch, 3, 3)

            rt = lie_metrics.as_repr(rt, self.a.repr_type) #size(batch, 3)
            zt = lie_metrics.as_repr(zt.unsqueeze(1), self.a.repr_type) #size(batch, 3)
            r0 = lie_metrics.as_repr(r0, self.a.repr_type) #size(batch, 3)

            rts.append(rt.unsqueeze(1))
            ts.append(t.unsqueeze(1))
            zts.append(zt.unsqueeze(1))
            r0s.append(r0.unsqueeze(1))
        
        batch["img"] = img.unsqueeze(1)
        batch["rt"] = torch.cat(rts, dim = 1)
        batch["ts"] = torch.cat(ts, dim = 1)
        batch["zts"] = torch.cat(zts, dim = 1)
        batch["r0s"] = torch.cat(r0s, dim = 1)

        return batch


    def train(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (1))
        ])

        train_dataset = dataset.load_symmetric_solids_dataset(split='train', transform=None)
        test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.a.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.a.batch_size)
        
        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = self.model
        model.train()
        model = model.to(device)
        
        optim = torch.optim.AdamW(self.model.params, lr=self.a.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.a.lr_decay_rate)
        
        for epoch in range(self.epochs):
            for batch_idx, (img, rot) in enumerate(train_loader):
                batch = self.get_flat_batch_train(img, rot)
                img = batch["img"].to(device)
                rt = batch["rt"].to(device)
                t = batch["t"].reshape((-1, 1)).to(device)

                mu = model(img, rt, t)
                ta = batch["zt"] if self.a.learn_noise else batch["r0"]
                loss = lie_metrics.distance_fn(ta, mu, self.a.loss_name)

                optim.zero_grad()
                loss.backward()
                optim.step()
            lr_scheduler.step()



def main():
    random_image = torch.rand(5, 3, 224, 224)
    rot = torch.randn((5, 3, 3))
    
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

    a = A()
    test = Testbed(a)
    test.get_flat_batch_train(random_image, rot, 10)




    return

if __name__ == "__main__":
    main()