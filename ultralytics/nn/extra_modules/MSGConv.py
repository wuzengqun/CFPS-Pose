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
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels,
                                   padding=kernel_size // 2, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# 定义的DualConv 类
class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, g=2):
        super(DualConv, self).__init__()
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.gc_bn = nn.BatchNorm2d(out_channels)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.pwc_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_data):
        gc_out = self.activation(self.gc_bn(self.gc(input_data)))
        pwc_out = self.activation(self.pwc_bn(self.pwc(input_data)))
        output = self.alpha * gc_out + (1 - self.alpha) * pwc_out
        return output

class MSGConv(nn.Module):
    # Multi-Scale Ghost Conv
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        print(f'MSGConv init: c1: {c1}, c2: {c2}, min_ch: {min_ch}, groups: {self.groups}')  # �����һ��
        self.s = s

        self.cv1 = Conv(c1, min_ch, k, s)
        # 使用 DualConv 替换 cv1
        # self.cv1 = DualConv(c1, min_ch, stride=s, g=2)

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        return x


class MSGAConv(nn.Module):  # MSGRConv
    # Multi-Scale Ghost Residual Conv
    def __init__(self, c1, c2, k=3, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        # print(f'MSGAConv init: c1: {c1}, c2: {c2}, min_ch: {min_ch}, groups: {self.groups}')  # �����һ��
        self.s = s

        self.convs = nn.ModuleList([])
        if s == 1:
            self.cv1 = Conv(c1, min_ch, 1, 1)
        if s == 2:
            self.cv1 = Conv(c1, min_ch, 3, 2)
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):

        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)

        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        out = torch.cat([x1, x2], dim=1)
        x = self.shortcut(x)
        out = self.conv1x1(out) + x
        return out



