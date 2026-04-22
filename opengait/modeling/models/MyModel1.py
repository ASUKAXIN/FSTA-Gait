import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from kornia.filters import spatial_gradient

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D, TCAM, MBTSA, MBTSB, MBTSC
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
import torch.nn.functional as F
from einops import rearrange

blocks_map = {
    '2d': BasicBlock2D,
    'MBTSA': MBTSA,
    'MBTSB': MBTSB,
    'MBTSC': MBTSC
}

class FSTAGait(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']

        blockA = blocks_map[mode[0]]
        blockB = blocks_map[mode[1]]
        blockC = blocks_map[mode[2]]

        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d':
            stride_1 = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            stride_1 = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        stride_2 = [
            [1, 1],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1]
        ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.layer1 = self.make_layer(BasicBlock2D, channels[0], stride_1[0], stride_2[0], blocks_num=layers[0], mode='2d')
        self.layer2 = self.make_layer(blockA, channels[1], stride_1[1], stride_2[1], blocks_num=layers[1], mode=mode[0])
        self.layer3 = self.make_layer(blockB, channels[2], stride_1[2], stride_2[2], blocks_num=layers[2], mode=mode[1])
        self.layer4 = self.make_layer(blockC, channels[3], stride_1[3], stride_2[3], blocks_num=layers[3], mode=mode[2])

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TCAM = TCAM(in_channels=channels[3], parts_num=[16])
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=[16]))

    def make_layer(self, block, planes, stride_1, stride_2, blocks_num, mode):

        if max(stride_1) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride_1,
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride_1),
                                           nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride_1],
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode in ['MBTSA', 'MBTSB', 'MBTSC']:
                # 对于 P3D 相关模式，需要将 stride_1 和 stride_2 合并为 3D stride
                # 假设 stride_1 处理空间维度，stride_2 处理时间维度
                if len(stride_1) == 2:  # [h_stride, w_stride]
                    # 将 2D stride 转换为 3D stride，时间维度 stride 为 1
                    stride_3d = [1, stride_1[0], stride_1[1]]
                else:  # 已经是 3D
                    stride_3d = stride_1

                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                              stride=stride_3d, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                # 其他模式使用默认的 3D 卷积
                if len(stride_1) == 2:
                    stride_3d = [1, stride_1[0], stride_1[1]]
                else:
                    stride_3d = stride_1

                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                              stride=stride_3d, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
        else:
            downsample = None

        layers = [block(self.inplanes, planes, stride_1=stride_1, stride_2=stride_2, downsample=downsample)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks_num):
            if mode == '2d':
                s_1 = [1, 1]
                s_2 = [1, 1, 1]
            else:
                s_1 = [1, 1] if len(stride_1) == 2 else [1, 1, 1]
                s_2 = [1, 1, 1]

            layers.append(block(self.inplanes, planes, stride_1=s_1, stride_2=s_2))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # NewHP
        outs = self.HPP(out4)
        outs = self.TCAM(outs)

        embed_1 = self.FCs(outs)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            # 训练时需用到的特征、标签、预测分数
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            # 用于可视化监控的数据（如中间帧图像）
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            # 推理时输出的核心特征（用于匹配/检索）
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
