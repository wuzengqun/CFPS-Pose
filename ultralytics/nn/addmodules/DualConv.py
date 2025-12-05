import torch.nn as nn


class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g=2):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)

# import torch
# import torch.nn as nn
#
# class DualConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, g=2):
#         super(DualConv, self).__init__()
#         # Group Convolution
#         self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
#         self.gc_bn = nn.BatchNorm2d(out_channels)
#         # Pointwise Convolution
#         self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#         self.pwc_bn = nn.BatchNorm2d(out_channels)
#         # Activation function
#         self.activation = nn.ReLU()
#         # Learnable weights
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, input_data):
#         gc_out = self.activation(self.gc_bn(self.gc(input_data)))
#         pwc_out = self.activation(self.pwc_bn(self.pwc(input_data)))
#         # Weighted sum
#         output = self.alpha * gc_out + (1 - self.alpha) * pwc_out
#         return output
