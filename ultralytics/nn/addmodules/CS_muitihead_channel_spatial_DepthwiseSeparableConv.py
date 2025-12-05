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

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        #self.qkv = Conv(dim, h, 1, act=False)
        #self.proj = Conv(dim, dim, 1, act=False)
        
        # 使用深度可分离卷积替代标准卷积
        self.qkv = DepthwiseSeparableConv(dim, h, kernel_size=1)
        self.proj = DepthwiseSeparableConv(dim, dim, kernel_size=1)
        #self.pe = DepthwiseSeparableConv(dim, dim, kernel_size=3, padding=1)
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


#class CS_Attention_DepthwiseSeparableConv(nn.Module):
#     def __init__(self, outchannels):
#         super(CS_Attention_DepthwiseSeparableConv, self).__init__()
#         self.spatial_attention = Attention(outchannels, attn_ratio=0.5, num_heads=max(1, outchannels // 128))
#         self.channnel_attention = SE_Block(outchannels, int(outchannels ** 0.5))
#
#     # def forward(self, x):       #串联
#     #     output = self.channnel_attention(x)
#     #     output = self.spatial_attention(output)
#     #
#     #     return output
#
#     # def forward(self, x):       #残差连接
#     #     residual = x  # 保存输入
#     #     output = self.channnel_attention(x)  # 通道注意力
#     #     output = self.spatial_attention(output)  # 空间注意力
#     #     return output + residual  # 残差连接
#
#     def forward(self, x):        # 加权求和
#         channel_output = self.channnel_attention(x)
#         spatial_output = self.spatial_attention(x)
#         return 0.5 * channel_output + 0.5 * spatial_output  # 加权求和
#
#     # def forward(self, x):        #并行
#     #     channel_output = self.channnel_attention(x)
#     #     spatial_output = self.spatial_attention(x)
#     #     return torch.cat((channel_output, spatial_output), dim=1)  # 按通道拼接
class CS_Attention_DepthwiseSeparableConv(nn.Module):
    def __init__(self, outchannels):
        super(CS_Attention_DepthwiseSeparableConv, self).__init__()
        self.spatial_attention = Attention(
            outchannels, attn_ratio=0.5, num_heads=max(1, outchannels // 128)
        )
        self.channel_attention = SE_Block(outchannels, int(outchannels ** 0.5))
        self.learnable_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的权重

    def forward(self, x):
        channel_output = self.channel_attention(x)
        spatial_output = self.spatial_attention(x)
        # 使用可学习的权重
        return self.learnable_weight * channel_output + (1 - self.learnable_weight) * spatial_output



