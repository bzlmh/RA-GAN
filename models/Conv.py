import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class GGC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2), theta_hidden_channels=32):
        super(GGC, self).__init__()
        self.activation = activation

        # globe convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        # 1x1 convolutional layer for local convolution
        self.diff_conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=stride, padding=0, bias=bias)

        # Theta network to compute dynamic gating parameter
        self.theta_net = nn.Sequential(
            nn.Conv2d(in_channels, theta_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(theta_hidden_channels, theta_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(theta_hidden_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        self._init_theta_weights()

    def _init_theta_weights(self):
        # Initialize weights of the theta network
        for m in self.theta_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Normal convolution output
        out_normal = self.conv(x)

        # Compute dynamic gating parameter (theta)
        theta = self.theta_net(x).mean()

        if math.fabs(theta - 0.0) < 1e-8:
            out_combined = out_normal
            grad_info = torch.zeros_like(out_normal)
        else:
            # Set differential convolution weights based on kernel differences
            with torch.no_grad():
                kernel_diff = self.conv.weight.sum(2, keepdim=True).sum(3, keepdim=True)
                self.diff_conv.weight = torch.nn.Parameter(kernel_diff)
            out_diff = self.diff_conv(x)

            out_combined = out_normal - theta * out_diff

        x, y = torch.chunk(out_combined, 2, 1)
        y = torch.sigmoid(y)
        x = self.activation(x) * y

        return x
class GateConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateConv, self).__init__()
        self.gate01 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        #print('input', input.shape)
        x = self.gate01(input)
        if self.activation is None:
            return x
        x, y = torch.chunk(x, 2, 1)
        y = torch.sigmoid(y)
        x = self.activation(x)
        x = x * y
        #print('output', x.shape)
        return x




class GateDeConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateDeConv, self).__init__()
        self.gate01 = GateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activation=activation)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='bilinear')
        x = self.gate01(x)
        return x
# import torch
# import torch.nn as nn
# import math
#
#
# class DynamicGradientGateConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True,
#                  activation=torch.nn.LeakyReLU(0.2), theta_hidden_channels=32):
#         super(DynamicGradientGateConv, self).__init__()
#         self.activation = activation
#
#
#         self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#
#         self.diff_conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=stride, padding=0, bias=bias)
#
#
#         self.theta_net = nn.Sequential(
#             nn.Conv2d(in_channels, theta_hidden_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(theta_hidden_channels, theta_hidden_channels, kernel_size=3, stride=1, padding=1),  # 额外的卷积层
#             nn.ReLU(),
#             nn.Conv2d(theta_hidden_channels, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#
#
#         self._init_theta_weights()
#
#     def _init_theta_weights(self):
#         for m in self.theta_net.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#
#         out_normal = self.conv(x)
#
#
#         theta = self.theta_net(x).mean()
#
#         if math.fabs(theta - 0.0) < 1e-8:
#
#             out_combined = out_normal
#             grad_info = torch.zeros_like(out_normal)
#         else:
#
#             with torch.no_grad():
#                 kernel_diff = self.conv.weight.sum(2, keepdim=True).sum(3, keepdim=True)
#                 self.diff_conv.weight = torch.nn.Parameter(kernel_diff)  # 将差分结果作为 diff_conv 的权重
#
#             out_diff = self.diff_conv(x)
#
#             out_combined = out_normal - theta * out_diff
#             grad_info = out_diff
#
#         x, y = torch.chunk(out_combined, 2, 1)
#         grad_x, grad_y = torch.chunk(grad_info, 2, 1)
#         y = torch.sigmoid(y)
#         x = self.activation(x)
#         x = x * y
#         return x
#
#
