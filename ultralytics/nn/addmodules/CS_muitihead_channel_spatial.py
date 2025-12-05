import torch
import torch.nn as nn
from thop import profile


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelsPool(nn.Module):
    def __init__(self):
        super(ChannelsPool, self).__init__()

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)), dim=1
        )


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
#
#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# class ChannelsPool(nn.Module):
#     def __init__(self):
#         super(ChannelsPool, self).__init__()
#
#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         return torch.cat((max_pool, avg_pool), dim=1)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, attn_ratio=0.5):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.key_dim = int(self.head_dim * attn_ratio)
#         self.scale = self.key_dim ** -0.5
#         nh_kd = self.key_dim * num_heads
#         h = dim + nh_kd * 2
#
#         # 使用 1x1 卷积将通道数调整回 dim
#         self.reduce_channels = nn.Conv2d(dim + 2, dim, kernel_size=1, stride=1, bias=False)
#         self.qkv = Conv(dim, h, 1, act=False)
#         self.proj = Conv(dim, dim, 1, act=False)
#         self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
#         self.pool = ChannelsPool()  # 使用 ChannelsPool 类
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W
#
#         # 通过 ChannelsPool 聚合 max 和 avg 的空间特征
#         pooled_features = self.pool(x)  # 维度为 (B, 2, H, W)
#
#         # 拼接原始特征与 pooled 特征
#         x = torch.cat([x, pooled_features], dim=1)  # 拼接后的维度为 (B, C + 2, H, W)
#
#         # 通过 reduce_channels 将通道数调整回 dim
#         x = self.reduce_channels(x)  # 调整后的维度为 (B, C, H, W)
#
#         # Q-K-V 计算
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
#             [self.key_dim, self.key_dim, self.head_dim], dim=2)
#
#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         # 注意力加权
#         x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
#         x = self.proj(x)
#         return x


class CS_Attention(nn.Module):
    def __init__(self, outchannels):
        super(CS_Attention, self).__init__()
        self.spatial_attention = Attention(outchannels, attn_ratio=0.5, num_heads=outchannels // 64)
        self.channnel_attention = SE_Block(outchannels, int(outchannels ** 0.5))

    def forward(self, x):
        output = self.channnel_attention(x)
        output = self.spatial_attention(output)

        return output