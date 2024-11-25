from .testbed import Testbed
import torch

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
            self.batch_size = 16
            self.epochs = 150
            self.init_lr = 1e-2
            self.lr_decay_rate = 0.1
            self.n_slices = 128

    a = A()
    test = Testbed(a)
    test.train()


if __name__ == "__main__":
    main()

