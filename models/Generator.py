import torch
import torch.nn as nn
from models.Conv import GGC, GateConv, GateDeConv
from models.ga import GA
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = GateConv(in_channels, 2 * in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = GateConv(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = GateConv(in_channels, out_channels, kernel_size=1, stride=strides)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # First convolution and activation
        out = self.conv2(out)        # Second convolution
        if not self.same_shape:
            x = self.conv3(x)        # Adjust input x if shapes do not match
        out = out + x                # Add input x to output
        return F.relu(out)           # Final activation


class RA_GAN(nn.Module):
    """
    Generator with Gate Convolutions
    """
    def __init__(self, input_c):
        super(RA_GAN, self).__init__()
        self.c = 64  # Number of feature channels

        # Gradient Gate Convolutional Layers (GGC)
        self.corase_a1_ = GGC(input_c, self.c, kernel_size=5, stride=1, padding=2)
        # Guide Attention (GA)
        self.ga1 = GA(self.c, self.c)
        self.corase_a1 = GGC(self.c, self.c, kernel_size=3, stride=2, padding=1)
        self.ga2 = GA(self.c, self.c)
        self.corase_a2 = GGC(self.c, self.c, kernel_size=3, stride=1, padding=1)
        self.ga3 = GA(self.c, self.c)
        self.corase_a3 = GGC(self.c, self.c, kernel_size=3, stride=1, padding=1)
        self.ga4 = GA(self.c, self.c)

        # Residual Blocks
        self.res1 = Residual(self.c, 2 * self.c, same_shape=False)
        self.res2 = Residual(self.c, 2 * self.c)
        self.res3 = Residual(self.c, 4 * self.c, same_shape=False)
        self.res4 = Residual(2 * self.c, 4 * self.c)
        self.res5 = Residual(2 * self.c, 8 * self.c, same_shape=False)
        self.res6 = Residual(4 * self.c, 8 * self.c)
        self.res7 = Residual(4 * self.c, 16 * self.c, same_shape=False)
        self.res8 = Residual(8 * self.c, 16 * self.c)

        # Projection Layers
        self.prj_5 = nn.Conv2d(256, 512, kernel_size=1)
        self.prj_4 = nn.Conv2d(128, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(64, 128, kernel_size=1)
        self.prj_2 = nn.Conv2d(64, 64, kernel_size=1)

        # Deconvolution Layers
        self.corase_c1s = GateDeConv(8 * self.c, 16 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c2s = GateDeConv(8 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c3s = GateDeConv(4 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c4s = GateDeConv(2 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c5s = GateConv(self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c6s = GateDeConv(self.c, 2, kernel_size=3, stride=1, padding=1, activation=torch.sigmoid)

    def forward(self, ori, ostu, sobel):
        img_input = torch.cat((ori, ostu, sobel), 1)  # Concatenate inputs
        y = self.corase_a1_(img_input)
        y, attention1 = self.ga1(y, y)
        y = self.corase_a1(y)

        y, attention2 = self.ga2(y, y)
        y = self.corase_a2(y)

        y, attention3 = self.ga3(y, y)
        y = self.corase_a3(y)

        y, attention4 = self.ga4(y, y)

        # Residual blocks and projections
        C2 = self.prj_2(y)
        y = self.res1(y)
        y = self.res2(y)

        C3 = self.prj_3(y)
        y = self.res3(y)
        y = self.res4(y)

        C4 = self.prj_4(y)
        y = self.res5(y)
        y = self.res6(y)
        C5 = self.prj_5(y)
        y = self.res7(y)
        y = self.res8(y)

        # GateDeConv Feature Pyramid(GFP)
        N5 = self.corase_c1s(y)
        P5 = C5 + N5

        N4 = self.corase_c2s(P5)
        P4 = C4 + N4

        N3 = self.corase_c3s(P4)
        P3 = C3 + N3

        N2 = self.corase_c4s(P3)
        P1 = C2 + N2

        # Final output
        bin_out = self.corase_c5s(P1)
        bin_out = self.corase_c6s(bin_out)

        return bin_out

