import torch
import torch.nn as nn
import math
from einops import rearrange
 
 
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # print(f'Conv init: c1: {c1}, c2: {c2}, k: {k}, s: {s}, g: {g}')  # �����һ��
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels, padding=kernel_size // 2, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# class BLMSGConv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
#         super().__init__()
#         self.groups = len(kernels)
#         min_ch = c2 // 2
#
#         self.cv1 = Conv(c1, min_ch, k, s)
#
#         self.cv2 = DepthwiseSeparableConv(min_ch, c2, 1)
#
#         self.convs = nn.ModuleList([])
#         for ks in kernels:
#             self.convs.append(DepthwiseSeparableConv(min_ch, min_ch, ks))
#
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = rearrange(x1, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
#         x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
#         x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
#         x = self.cv2(x2)
#         return x


class BLMSGConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2

        # 确保 min_ch 是 self.groups 的倍数
        assert min_ch % self.groups == 0, "min_ch must be divisible by self.groups"

        self.cv1 = Conv(c1, min_ch, k, s)     #将输入特征图的通道数从 c1 压缩到 min_ch
        self.cv2 = DepthwiseSeparableConv(min_ch, c2, 1)
        # self.cv2 = DepthwiseSeparableConv(min_ch * self.groups, c2, 1)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(min_ch // self.groups, min_ch // self.groups, ks) for ks in kernels])
        # self.convs = nn.ModuleList([
        #     DepthwiseSeparableConv(min_ch, min_ch, ks) for ks in kernels
        # ])


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = rearrange(x1, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x2 = torch.cat([self.convs[i](x2[..., i]) for i in range(self.groups)], dim=1)
        # x2 = torch.cat([conv(x1) for conv in self.convs], dim=1)
        x = self.cv2(x2)
        return x

    

       

 

