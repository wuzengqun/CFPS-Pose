import torch
import torch.nn as nn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostGroupedConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ghost = max(1, c2 // 4)  # 使用更小的通道数减少计算量
        self.grouped_conv = nn.Conv2d(c1, c_ghost, k, s, autopad(k, p), groups=max(1, c_ghost // g), dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c_ghost)
        self.act1 = nn.SiLU() if act else nn.Identity()

        self.ghost_conv = nn.Conv2d(c_ghost, c_ghost, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c_ghost)
        self.act2 = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.grouped_conv(x)))
        x_ghost = self.act2(self.bn2(self.ghost_conv(x)))
        return torch.cat([x, x_ghost], dim=1)

class GGConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = max(self.groups, (c2 // 2) // self.groups * self.groups)  # 确保 min_ch 与 self.groups 匹配

        self.cv1 = Conv(c1, min_ch, k, s)
        self.cv2 = GhostGroupedConv(min_ch * self.groups, c2, 1)
        self.convs = nn.ModuleList([
            GhostGroupedConv(min_ch, min_ch, ks) for ks in kernels
        ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat([conv(x1) for conv in self.convs], dim=1)
        x = self.cv2(x2)
        return x
