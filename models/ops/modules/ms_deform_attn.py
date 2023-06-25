# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 采样点的坐标偏移，每个query在每个注意力头和每个特征层都需要采样n_points个.
        # 由于x\y坐标都有对应的偏移量，因此要*2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每个query对应的所有采样点的注意力权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 线性变换得到value
        self.value_proj = nn.Linear(d_model, d_model)
        # 最后经过线性变换得到输出结果
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        """
        1.将输入input_flatten(对于Encoder就是由backbone输出的特征图变换而来；对于decoder就是Encoder输出的memory)
        通过变换矩阵得到value，同时将padding的部分用0填充.
        2.将query(对于Encoder就是特征图本身加上position&scale-level embedding; 对于decoder就是self-attention的输出加上position embedding结果，
        2-stage时这个position embedding是有Encoder预测的top-k proposal boxes进行positon embedding得来；而1-stage是预设的embedding)分别通过两个
        全连接层得到采样点对应的坐标偏移和注意力权重(注意力权重会进行归一化).
        3.根据参考点(reference points:对于Decoder来说，2-stage时时Encoder预测的top-k proposal boxes;1-stage时是由预设的query embedding经过全连接
        层得到。两种情况下最终都经过了sigmoid函数归一化; 对于Encoder来说，就是各特征点在所有特征层对应的归一化中心坐标)坐标和预测坐标偏移得到采样点坐标.
        4.由采样点坐标在value中插值采样出对应的特征向量，然后施加注意力权重，最后将这个结果经过全链接层得到输出结果
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 这个值需要是所有特征层特征点的数量
        # input_spatial_shapes 每一层的高宽
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # (N, Len_in, d_model=256)
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            # 将原图padding的部分用0填充
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # (N, Len_in, 8, 32) 拆分成8个注意力头部对应的维度
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # (N, Len_in, 8, 4, 4, 2)预测采样点的坐标偏移
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # (N, Len_in, 8, 4*4) 预测采样点对应的注意力权重
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # (N, Len_in, 8, 4, 4) 每个query在每个注意力头部内，每个特征层都采样里个特征点,即16(4*4)个采样，这16个对应的权重进行归一化
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        '''计算采样点的坐标(以下参考点的坐标已归一化)'''
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # (4,2)其中每个是(w, h)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # (N, Len_q, 1, n_levels=4,1,2) + (N, Len_q, 8, n_levels=4,4,2)/(1,1,1,n_levels=4,1,2)
            # 对坐标偏移量使用对应层特征图的宽高进行归一化，然后家在参考点的坐标上得到采样点的坐标
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # output = MSDeformAttnFunction.apply(
        #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
