import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class Backbone(nn.Module):
    """
    Args:
        image_shape (list, tuple): shape of input image e.g. [1 x 3 x 224 x 224]
        feat_dim (int): output dimension of convolution layer e.g. 128
        dim (int): output dimension of backbone e.g. 512
        resnet_depth (int): e.g. 34, 50
    """
    def __init__(self, image_shape, feat_dim, dim, resnet_depth):
        super(Backbone, self).__init__()
        self.image_shape = image_shape
        self.feat_dim = feat_dim
        self.dim = dim
        self.resnet_depth = resnet_depth

        if self.resnet_depth == 34:
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.resnet_depth == 50:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2]) # Remove FC layer and AvgPool
        
        # get the output size of the backbone
        dummy_input = torch.randn(self.image_shape)
        dummy_temp = self.backbone(dummy_input)
        _, conv_in_dim, output_w, output_h = dummy_temp.shape
        linear_in_dim = self.feat_dim * output_w * output_h

        # Initialize some layers for the output of backbone
        self.conv = nn.Conv2d(in_channels=conv_in_dim, 
                              out_channels=self.feat_dim,
                              kernel_size=(1, 1), 
                              stride=(1, 1))

        self.linear = nn.Linear(in_features=linear_in_dim,
                                out_features=self.dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = x.reshape((x.shape[0], -1))
        x = self.linear(F.leaky_relu(x))
        return x

class PosEmbed(nn.Module):
    """
    Args:
        in_feat_dim (int): feature dimension of input, e.g. t:1, rt: 3
        embed_dim (int): embedding-dim of each element of feature,
                    so the input of linear layer is in_feat_dim * embed_dim
        dim (int): output dimension
        shift: whether to use bias
    """
    def __init__(self, in_feat_dim, embed_dim, dim, shift=True):
        super(PosEmbed, self).__init__()
        self.in_feat_dim = in_feat_dim
        self.embed_dim = embed_dim
        self.dim = dim                 
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)

        # Generate log-scale arithmetic sequence, length=half_dim 1.0~1.0e-4
        self.log_scale_seq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        self.mlp = nn.Linear(self.in_feat_dim * self.embed_dim, self.dim, bias=shift)

        # debug
        self.linear = nn.Linear(self.in_feat_dim, self.dim)

    def forward(self, x):
        # device = x.device
        # x = x.unsqueeze(-1)
        # # print(x.shape)
        # emb = x * self.log_scale_seq.to(device)
        # # print(emb.shape)
        # emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # emb = emb.view((emb.shape[0], -1)) # Reshaping (N, -1)
        # # print(emb.shape)
        # emb = self.mlp(emb)
        # return emb
        return self.linear(x)

class FourierMlpBlock(nn.Module):
    """
    The FourierMlpBlock takes x and c as input, output the feature with out_dim
    
    Args:
        x_dim (int): feature dimension of x, e.g. 512
        c_dim (int): feature dimension of c, e.g. 256
        out_dim (int): output dimension, e.g. 256
    """
    def __init__(self, x_dim, c_dim, out_dim):
        super(FourierMlpBlock, self).__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.c_dim, self.out_dim * 2)
        self.linear2 = nn.Linear(self.x_dim, self.out_dim)
        self.linear3 = nn.Linear(self.out_dim, self.out_dim)
        self.linear4 = nn.Linear(self.x_dim, self.out_dim)

        # debug
        self.linearx = nn.Linear(self.x_dim, self.out_dim)
        self.linearc = nn.Linear(self.c_dim, self.out_dim)

    # The input dimensional of x_in and c are the same??
    def forward(self, x_in, c):
        # c = self.linear1(F.silu(c))
        # a, b = c.chunk(2, dim=-1)
        # x = self.linear2(F.leaky_relu(x_in))
        # x = a * torch.cos(x * torch.pi) + b * torch.sin(x * torch.pi)
        # x = self.linear3(x)
        # return x + self.linear4(x_in)
        return F.leaky_relu(self.linearx(x_in) + self.linearc(c))

class Head(nn.Module):
    """
    The head takes current pose as input (x_i), and output the score s(x_i, sigma_i)
    """
    def __init__(self, in_dim, dim, n_layers = 1, block_type = 'MlpBlock', activ = 'leaky_relu'):
        """
        Args:
        in_dim (int): input of the head
        dim (int): dimension of output of head
        n_layers (int): # of MLP layers
        """

        super(Head, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.n_layers = n_layers
        self.posEmbedT = PosEmbed(1, 256, 256, True)
        self.posEmbedRt = PosEmbed(self.in_dim, 256, 256, False)

        self.mlpBlocks = nn.ModuleList()
        for _ in range(n_layers):
            self.mlpBlocks.append(nn.ModuleList([
                FourierMlpBlock(256, 512, 256),  # image-condition
                FourierMlpBlock(256, 256, 256)   # time-condition
            ]))

        self.linear = nn.Linear(256, self.dim)

    def broadcast_batch(self, x, bs):
        batch_size = x.shape[0]
        slices = bs // batch_size

        x = x.repeat(slices, 1)
        # print(x.shape)
        # x = x.repeat(slices, 1).reshape((-1,) + (x.shape[1], ))
        return x


    def forward(self, img_feat, rt, t):
        """
        Args:
        img_feat: conditional signal - image (x0 in original code), shape (batch, feat_dim)
        rt: rotation matrix (use so3 representation), shape (batch * n_slices, 3), e.g., (4096, 3)
        t: conditional signal - time index, shape (batch * n_slices), e.g., (4096)

        Returns:
        model output: shape (batch * n_slices, 3) 
        """
        img_feat = self.broadcast_batch(img_feat, rt.shape[0])
        # print(img_feat[:512, 0])
        # torch.set_printoptions(threshold=torch.inf)

        t = t.float()
        t = self.posEmbedT(t)
        x = self.posEmbedRt(rt)

        for blockImg, blockT in self.mlpBlocks:
            x = blockT(blockImg(x, img_feat), t)

        x = self.linear(F.leaky_relu(x))

        return x

class Model(nn.Module):
    def __init__(self, 
                 in_dim=6, 
                 out_dim=6,
                 image_shape=[1, 3, 224, 224],
                 resnet_depth=34,
                 mlp_layers=1,
                 fourier_block=True,
                 activ_fn='leaky_relu'):
        
        super(Model, self).__init__()
        self.backbone = Backbone(image_shape, feat_dim=128, dim=512, resnet_depth=resnet_depth)
        self.head = Head(in_dim=in_dim, dim=out_dim, n_layers=mlp_layers, block_type=fourier_block, activ=activ_fn)
    
    """
    Args:
    img: conditional signal - image, shape (batch, 3, image_size, image_size), e.g., (16, 3, 224, 224)
    rt: rotation matrix (use so3 representation), shape (batch * n_slices, 3), e.g., (2048, 3)
    t: conditional signal - time index, shape (batch * n_slices), e.g., (2048)

    Returns:
    model output: shape (batch * n_slices, 3) 
    """
    def forward(self, img, rt, t):
        x = self.backbone(img)
        mu = self.head(x, rt, t)
        return mu

def main():
    # Test Module1: backbone block
    # Initialize a batch of image_shape, batch_size = 2 
    image_shape = [10, 3, 224, 224]
    img = torch.randn(image_shape)

    backbone = Backbone(image_shape=image_shape, feat_dim=64, dim=512, resnet_depth=34)
    output = backbone(img)
    
    # # Expected [1, 512]
    # print(output.shape)

    # # Test Module2: PosEmbed block
    # posembed = PosEmbed(3, 256, 256)
    # emb = torch.randn((10, 3))
    # output = posembed(emb)
    # print(output.shape)

    # # Test Module3: FourierMlpBlock 
    # mlp = FourierMlpBlock(256, 256, 7)
    # input = torch.randn((10, 256))
    # output = mlp(input, input)
    # print(output.shape)

    # # Test Module4: Head
    # head = Head(256, n_layers=1)
    # img = torch.randn((10, 512))
    # t = torch.randn((10, 1))
    # x = torch.randn((10, 3))
    # output_x = head(img, x, t)
    # print(output_x.shape)

    # # Test Module5: Model
    # image_shape = (10, 3, 224, 224)
    # model = Model(in_dim=3, out_dim=3, image_shape=image_shape, resnet_depth=50, mlp_layers=1, fourier_block=True, activ_fn='leaky_relu')
    # img = torch.randn(image_shape)
    # t = torch.randn((10, 1))
    # x = torch.randn((10, 3))
    # output = model(img, rt=x, t=t)
    # print(model)
    # print(output.shape)


if __name__ == '__main__':
    main()