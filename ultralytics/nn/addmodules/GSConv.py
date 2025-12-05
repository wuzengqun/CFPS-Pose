import torch
from torch import nn
import math
from einops import rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):  # 添加 d 参数
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True):
        super().__init__()
        # Number of primary and ghost features
        c_ghost = c2 // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c_ghost, k, s, autopad(k, p), dilation=d, bias=False),
            nn.BatchNorm2d(c_ghost),
            nn.SiLU() if act else nn.Identity()
        )
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(c_ghost, c_ghost, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_ghost),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        x_primary = self.primary_conv(x)
        x_ghost = self.ghost_conv(x_primary)
        return torch.cat([x_primary, x_ghost], dim=1)



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1,act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)

# class GSConv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = GhostConv(c1, c_, k, s, None, d, act)  # 使用 GhostConv 替代普通卷积
#         self.cv2 = GhostConv(c_, c_, 5, 1, None, d, act)  # 使用 GhostConv 替代普通卷积
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # shuffle
#         b, n, h, w = x2.data.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#         return torch.cat((y[0], y[1]), 1)
