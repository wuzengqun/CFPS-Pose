# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "DSConv",
    "ConvBNAct"
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


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True) -> None:
        super().__init__()
        
        self.dwconv = DWConv(c1, c1, 3)
        self.pwconv = Conv(c1, c2, 1)
    
    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

def get_norm(norm_type, out_channels, inplace=True):
    """
    根据指定的规范化类型返回对应的规范化层。

    参数:
        norm_type (str): 规范化类型（例如 'bn'）。
        out_channels (int): 输出通道数，用于 BatchNorm 等层。
        inplace (bool): 是否进行原地操作（默认True）。

    返回:
        nn.Module: 返回指定类型的规范化层。
    """
    if norm_type == 'bn':
        return nn.BatchNorm2d(out_channels)  # 返回 Batch Normalization 层
    elif norm_type == 'ln':
        return nn.LayerNorm(out_channels)  # 返回 Layer Normalization 层
    elif norm_type == 'gn':
        return nn.GroupNorm(32, out_channels)  # 返回 Group Normalization 层
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def get_activation(activation_type, inplace=True):
    """
    根据指定的激活函数类型返回对应的激活函数。

    参数:
        activation_type (str): 激活函数类型（例如 'relu', 'sigmoid', 'tanh' 等）。
        inplace (bool): 是否进行原地操作（默认True）。

    返回:
        nn.Module: 返回指定类型的激活函数。
    """
    if activation_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'leakyrelu':
        return nn.LeakyReLU(inplace=inplace)
    elif activation_type == 'swish' or activation_type == 'silu':
        return nn.SiLU(inplace=inplace)  # SiLU激活函数（Swish）
    else:
        raise ValueError(f"未知的激活函数类型: {activation_type}")

class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self,
            in_channels,
            out_channels,
            ksize,
            stride=1,
            groups=1,
            bias=False,
            act='silu',
            norm='bn',
            reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


# class RepConv(nn.Module):
#     """
#     RepConv is a basic rep-style block, including training and deploy status.
#
#     This module is used in RT-DETR.
#     Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     """
#
#     # default_act = nn.SiLU()  # default activation  按照作者的意思，在较大模型中使用silu，科研带来更好的收益
#     default_act = nn.ReLU()     # 这里是小模型，所以先用relu试一下
#
#     def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
#         """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
#         super().__init__()
#         assert k == 3 and p == 1
#         self.g = g
#         self.c1 = c1
#         self.c2 = c2
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#         self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
#         self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
#         self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
#
#     def forward_fuse(self, x):
#         """Forward process."""
#         return self.act(self.conv(x))
#
#     def forward(self, x):
#         """Forward process."""
#         id_out = 0 if self.bn is None else self.bn(x)
#         return self.act(self.conv1(x) + self.conv2(x) + id_out)
#
#     def get_equivalent_kernel_bias(self):
#         """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
#         kernelid, biasid = self._fuse_bn_tensor(self.bn)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
#
#     @staticmethod
#     def _pad_1x1_to_3x3_tensor(kernel1x1):
#         """Pads a 1x1 tensor to a 3x3 tensor."""
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
#
#     def _fuse_bn_tensor(self, branch):
#         """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, Conv):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         elif isinstance(branch, nn.BatchNorm2d):
#             if not hasattr(self, "id_tensor"):
#                 input_dim = self.c1 // self.g
#                 kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.c1):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std
#
#     def fuse_convs(self):
#         """Combines two convolution layers into a single layer and removes unused attributes from the class."""
#         if hasattr(self, "conv"):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv = nn.Conv2d(
#             in_channels=self.conv1.conv.in_channels,
#             out_channels=self.conv1.conv.out_channels,
#             kernel_size=self.conv1.conv.kernel_size,
#             stride=self.conv1.conv.stride,
#             padding=self.conv1.conv.padding,
#             dilation=self.conv1.conv.dilation,
#             groups=self.conv1.conv.groups,
#             bias=True,
#         ).requires_grad_(False)
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__("conv1")
#         self.__delattr__("conv2")
#         if hasattr(self, "nm"):
#             self.__delattr__("nm")
#         if hasattr(self, "bn"):
#             self.__delattr__("bn")
#         if hasattr(self, "id_tensor"):
#             self.__delattr__("id_tensor")

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
