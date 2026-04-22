import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class TCAM(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TCAM, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        self.conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1)
        )

        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
        x:   [N, C, T, P]   (你的S就是T)
        ret: [N, C, P]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [P, N, C, T]
        feature = x.split(1, 0)
        x1 = x.view(-1, c, s)  # [P*N, C, T]

        logist = torch.cat([self.conv3x1(_.squeeze(0)).unsqueeze(0) for _ in feature], 0)  # [P,N,C,T]
        scores = torch.sigmoid(logist)

        features = self.avg_pool3x1(x1) + self.max_pool3x1(x1)
        features = features.view(p, n, c, s)  # [P,N,C,T]

        channel_weights = self.channel_attention(x1).unsqueeze(-1)
        channel_weights = channel_weights.view(p, n, c, 1)  # [P,N,C,1]

        ret = (features * scores * channel_weights) + x

        ret = self.TP(ret, dim=-1)[0]

        ret = ret.permute(1, 2, 0).contiguous()

        return ret


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]  # 输入形状：n=批次大小，c_in=输入通道，p=parts_num
            out: [n, c_out, p]  # 输出形状：c_out=输出通道
        """
        # 维度调整：[n, c_in, p] → [p, n, c_in]，确保内存连续
        x = x.permute(2, 0, 1).contiguous()

        if self.norm:
            # 对权重在输入通道维度（dim=1）做L2归一化，再做矩阵乘法
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            # 直接用原始权重做矩阵乘法
            out = x.matmul(self.fc_bin)

        # 维度调整回 [n, c_out, p]
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num  # 局部特征的数量（如分箱数、人体部位数）
        self.class_num = class_num  # 分类任务的类别数（如行人ID数）
        self.norm = norm  # 是否对特征和权重进行L2归一化
        # 分离式分类权重：为每个局部特征分配独立的全连接权重矩阵
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num))  # 形状：[p, c_in, class_num]
        )
        # 定义BN层：两种模式（并行/分离）
        if parallel_BN1d:
            # 并行BN：将所有局部特征拼接后用一个BN层处理
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            # 分离BN：为每个局部特征分配独立的BN层
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d  # 标记BN模式

    def forward(self, x):
        """
            x: 输入特征，形状为 [n, c, p]
                n: 批次大小，c: 输入通道数，p: 局部特征数量（parts_num）
            返回值：
                feature: 经过BN和可能的归一化后的特征，形状 [n, c, p]
                logits: 每个局部特征的分类得分，形状 [n, class_num, p]
        """
        # 第一步：对局部特征进行BN处理（两种模式）
        if self.parallel_BN1d:
            # 并行BN：拼接所有局部特征后统一归一化
            n, c, p = x.size()
            x = x.view(n, -1)  # 形状：[n, c*p]（拼接所有局部特征的通道）
            x = self.bn1d(x)  # 对拼接后的特征做BN
            x = x.view(n, c, p)  # 恢复形状：[n, c, p]
        else:
            # 分离BN：每个局部特征独立归一化
            # 将x按局部特征维度（p）拆分，每个子特征形状为 [n, c, 1]
            # 对每个子特征应用对应的BN层，再拼接回 [n, c, p]
            x = torch.cat([bn(_x) for _x, bn in zip(x.split(1, 2), self.bn1d)], 2)

        # 第二步：特征维度调整与可能的归一化
        feature = x.permute(2, 0, 1).contiguous()  # 形状：[p, n, c]（局部特征维度提前）
        if self.norm:
            # 对特征和分类权重做L2归一化（常用于度量学习，增强特征判别性）
            feature = F.normalize(feature, dim=-1)  # 特征在通道维度归一化：[p, n, c]
            logits = feature.matmul(F.normalize(self.fc_bin, dim=1))  # 分类得分计算
        else:
            # 直接用原始特征和权重计算分类得分
            logits = feature.matmul(self.fc_bin)  # 矩阵乘法：[p, n, c] × [p, c, class_num] → [p, n, class_num]

        # 第三步：调整输出维度，与输入格式对齐
        return (
            feature.permute(1, 2, 0).contiguous(),  # 特征形状：[n, c, p]
            logits.permute(1, 2, 0).contiguous()  # 分类得分形状：[n, class_num, p]
        )

class FocalConv2d(nn.Module):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """

    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2 ** self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """

    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
        [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w + 2 * width_p)
        w_right = torch.clamp(w_right, min=0., max=w + 2 * width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


'''
Modifed from https://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/components/units
'''


class Graph():
    """
    # Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
    """

    def __init__(self, joint_format='coco', max_hop=2, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.joint_format == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([5, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14])
            parts = [
                np.array([5, 7, 9]),  # left_arm
                np.array([6, 8, 10]),  # right_arm
                np.array([11, 13, 15]),  # left_leg
                np.array([12, 14, 16]),  # right_leg
                np.array([0, 1, 2, 3, 4]),  # head
            ]

        elif self.joint_format == 'coco-no-head':
            num_node = 12
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1),
                             (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                             (6, 8), (8, 10), (7, 9), (9, 11)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([3, 1, 0, 2, 4, 0, 6, 8, 10, 7, 9, 11])
            parts = [
                np.array([0, 2, 4]),  # left_arm
                np.array([1, 3, 5]),  # right_arm
                np.array([6, 8, 10]),  # left_leg
                np.array([7, 9, 11])  # right_leg
            ]

        elif self.joint_format == 'alphapose' or self.joint_format == 'openpose':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            raise ValueError('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD


class TemporalBasicBlock(nn.Module):
    """
        TemporalConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=0, get_res=False,
                 tcn_stride=False):
        super(TemporalBasicBlock, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size, 1), (stride, 1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


class TemporalBottleneckBlock(nn.Module):
    """
        TemporalConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4, get_res=False,
                 tcn_stride=False):
        super(TemporalBottleneckBlock, self).__init__()
        tcn_stride = False
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        if get_res:
            if tcn_stride:
                stride = 2
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2, 1)),
                nn.BatchNorm2d(channels),
            )
            tcn_stride = True
        else:
            if not residual:
                self.residual = lambda x: 0
            elif stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, (2, 1)),
                    nn.BatchNorm2d(channels),
                )
                tcn_stride = True

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if tcn_stride:
            stride = 2
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size, 1), (stride, 1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)
        return x


class SpatialGraphConv(nn.Module):
    """
        SpatialGraphConv_Basic_Block
        Arxiv: https://arxiv.org/abs/1801.07455
        Github: https://github.com/yysijie/st-gcn
    """

    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)

    def forward(self, x, A):
        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v).contiguous()

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x


class SpatialBasicBlock(nn.Module):
    """
        SpatialGraphConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=0):
        super(SpatialBasicBlock, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x


class SpatialBottleneckBlock(nn.Module):
    """
        SpatialGraphConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=4):
        super(SpatialBottleneckBlock, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        return x


class SpatialAttention(nn.Module):
    """
    This class implements Spatial Transformer. 
    Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
    """

    def __init__(self, in_channels, out_channel, A, num_point, dk_factor=0.25, kernel_size=1, Nh=8, num=4, stride=1):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = int(dk_factor * out_channel)
        self.dv = int(out_channel)
        self.num = num
        self.Nh = Nh
        self.num_point = num_point
        self.A = A[0] + A[1] + A[2]
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                  stride=stride,
                                  padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)


from einops import rearrange


class ParallelBN1d(nn.Module):
    def __init__(self, parts_num, in_channels, **kwargs):
        super(ParallelBN1d, self).__init__()
        self.parts_num = parts_num
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, p]
        '''
        x = rearrange(x, 'n c p -> n (c p)')
        x = self.bn1d(x)
        x = rearrange(x, 'n (c p) -> n c p', p=self.parts_num)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=dilation, dilation=dilation, bias=False
    )

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride_1=1, stride_2=1,downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride_1)
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride_1),
                norm_layer2d(planes),
                nn.ReLU(inplace=True)
            )
        )
        self.bn1 = norm_layer2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn = norm_layer3d(planes)

        self.downsample = downsample
        self.stride = stride_1

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.shortcut3d(out)
        out = self.sbn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockP3D, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride),
                norm_layer2d(planes),
                nn.ReLU(inplace=True)
            )
        )

        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes),
                norm_layer2d(planes),
            )
        )

        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn = norm_layer3d(planes)

        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=[1, 1, 1], downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert stride[0] in [1, 2, 3]
        if stride[0] in [1, 2]:
            tp = 1
        else:
            tp = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=[tp, 1, 1], bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1], bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MBTSA(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride_1=1, stride_2=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MBTSA, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d

        # 确保 stride_1 和 stride_2 是合适的格式
        if isinstance(stride_1, int):
            stride_1 = [stride_1, stride_1]  # 空间 stride [h, w]
        if isinstance(stride_2, int):
            stride_2 = [stride_2, stride_2, stride_2]  # 3D stride [t, h, w]

        width = int(planes * (base_width / 64.)) * groups

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=False)

        # 1*1*1
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer3d(planes)

        # 3*3*3
        if len(stride_2) == 3:
            conv2_stride = stride_2
        else:
            # 假设 stride_2 处理时间维度，stride_1 处理空间维度
            conv2_stride = [stride_2[0] if len(stride_2) > 0 else 1,
                            stride_1[0], stride_1[1]]
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=conv2_stride, padding=1, bias=False)
        self.bn2 = norm_layer3d(planes)

        # 1*3*3
        self.conv3 = SetBlockWrapper(nn.Sequential(
            conv3x3(planes, planes, stride_1, dilation),
            norm_layer2d(planes * self.expansion),
            nn.ReLU(inplace=True)
        ))

        # 3*1*1
        self.conv4 = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.bn3 = norm_layer3d(planes)

        self.downsample = downsample
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        x2 = self.conv3(x)
        x3 = self.conv4(x2)
        x3 = self.bn3(x3)
        x4 = x2 + x3

        out = x1 + x4
        out = self.relu(out)
        out = F.dropout(out, p=0.2)

        return out

class MBTSB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride_1=1, stride_2=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MBTSB, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d

        # 确保 stride_1 和 stride_2 是列表格式
        if isinstance(stride_1, int):
            stride_1 = [stride_1, stride_1]
        if isinstance(stride_2, int):
            stride_2 = [stride_2, stride_2, stride_2]

        width = int(planes * (base_width / 64.)) * groups

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=False)

        # 1*1*1
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer3d(planes)

        # 3*3*3
        if len(stride_2) == 3:
            conv2_stride = stride_2
        else:
            conv2_stride = [stride_2[0], 1, 1]  # 假设 stride_2 的第一个元素是时间 stride
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=conv2_stride, padding=1, bias=False)
        self.bn2 = norm_layer3d(planes)

        # 1*3*3
        self.conv3 = SetBlockWrapper(nn.Sequential(
            conv3x3(planes, planes, stride_1, dilation),
            norm_layer2d(planes * self.expansion),
            nn.ReLU(inplace=True)
        ))

        # 3*1*1
        self.conv4 = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.bn3 = norm_layer3d(planes)

        self.downsample = downsample
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        x2 = self.conv3(x)
        x3 = self.conv4(x2)
        x3 = self.bn3(x3)

        out = x1 + x3
        out = self.relu(out)
        out = F.dropout(out, p=0.2)

        return out


class MBTSC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride_1=1, stride_2=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MBTSC, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d

        # 确保 stride_1 和 stride_2 是合适的格式
        if isinstance(stride_1, int):
            stride_1 = [stride_1, stride_1]  # 空间 stride [h, w]
        if isinstance(stride_2, int):
            stride_2 = [stride_2, stride_2, stride_2]  # 3D stride [t, h, w]

        width = int(planes * (base_width / 64.)) * groups

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.relu = nn.ReLU(inplace=False)

        # 1*1*1 卷积
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer3d(planes)

        # 3*3*3 卷积 - 使用 stride_2 作为 3D stride
        # 如果 stride_2 是 3D，直接使用；否则构建合适的 3D stride
        if len(stride_2) == 3:
            conv2_stride = stride_2
        else:
            # 假设 stride_2 处理时间维度，stride_1 处理空间维度
            conv2_stride = [stride_2[0] if len(stride_2) > 0 else 1,
                            stride_1[0], stride_1[1]]

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=conv2_stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer3d(planes)

        # 1*3*3 卷积 - 使用 stride_1 作为空间 stride
        self.conv3 = SetBlockWrapper(nn.Sequential(
            conv3x3(planes, planes, stride_1, dilation),
            norm_layer2d(planes * self.expansion),
        ))

        # 3*1*1 卷积 - 使用 stride_1 和 stride_2 的组合
        # 时间维度使用 stride_2，空间维度使用 stride_1
        if len(stride_2) == 3:
            conv4_stride_t = stride_2[0]
        else:
            conv4_stride_t = stride_2[0] if len(stride_2) > 0 else 1

        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1),
                               stride=(conv4_stride_t, stride_1[0], stride_1[1]),
                               padding=(1, 0, 0), bias=False)
        self.bn3 = norm_layer3d(planes)

        self.downsample = downsample
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x3 = self.bn3(x3)

        out = x1 + x2 + x3
        out = self.relu(out)
        out = F.dropout(out, p=0.2)

        return out

class TransformerFeatureFusion(nn.Module):
    def __init__(self,in_channel,num_heads=8,num_layers=1,dropout=0.1):
        super(TransformerFeatureFusion, self).__init__()
        # 定义transformer层
        self.transformer = nn.Transformer(d_model=in_channel,nhead=num_heads,num_encoder_layers=num_layers,dropout=dropout)

    def forward(self,x):
        #调整输入的张量形状[p, n, c]
        x0 = x.permute(2,0,1)

        out = self.transformer(x0,x0)
        out = out.permute(1,2,0)
        out = torch.add(x,out)
        # out = torch.cat([x, out],dim=1)
        return out


class SpatialCrossAttention(nn.Module):
    """
    Spatial-Transformer：基于交叉注意力的空间特征增强
    - q：来自Initial Block（输入维度：[n, q_ch, s, h_q, w_q]）
    - k/v：来自H3DA（layer2输出，维度：[n, kv_ch, s, h_kv, w_kv]）
    核心：将q的空间维度下采样匹配k/v，通过交叉注意力融合空间关联特征
    """

    def __init__(self, q_in_ch=64, kv_in_ch=128, d_model=64, num_heads=8, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        # 1. 通道投影：将q/k/v映射到统一的d_model维度（适配多头注意力）
        self.q_proj = nn.Conv3d(q_in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)  # q通道投影
        self.k_proj = nn.Conv3d(kv_in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)  # k通道投影
        self.v_proj = nn.Conv3d(kv_in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)  # v通道投影

        # 2. 空间下采样：将Initial Block的空间维度（h=64, w=44）匹配H3DA（h=32, w=22）
        self.q_downsample = nn.Conv3d(d_model, d_model, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        # 3. 多头交叉注意力（核心组件）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False
        )

        # 4. 输出投影与融合：将注意力特征投影回H3DA的通道数，便于残差融合
        self.out_proj = nn.Conv3d(d_model, kv_in_ch, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.norm = nn.BatchNorm3d(kv_in_ch)  # 稳定训练
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, q, kv):
        # q：Initial Block输出 → [32, 64, 30, 64, 44]（n=32, q_ch=64, s=30, h_q=64, w_q=44）
        # kv：H3DA输出 → [32, 128, 30, 32, 22]（n=32, kv_ch=128, s=30, h_kv=32, w_kv=22）
        n, _, s, _, _ = q.size()

        # 步骤1：q预处理（通道投影 + 空间下采样）
        q_proj = self.q_proj(q)  # [32, 64, 30, 64, 44] → [32, d_model, 30, 64, 44]
        q_down = self.q_downsample(q_proj)  # [32, d_model, 30, 32, 22]（空间维度匹配kv）

        # 步骤2：k/v预处理（仅通道投影，空间维度已匹配）
        k_proj = self.k_proj(kv)  # [32, d_model, 30, 32, 22]
        v_proj = self.v_proj(kv)  # [32, d_model, 30, 32, 22]

        # 步骤3：调整维度为MultiheadAttention输入格式 → [seq_len, batch_size, embed_dim]
        # 空间维度展平为seq_len（h*w=32*22=704），批次维度合并时序（s*n=30*32=960）
        q_flat = q_down.permute(2, 0, 1, 3, 4)  # [s, n, d_model, h, w] → [30, 32, d_model, 32, 22]
        q_flat = q_flat.reshape(s, n, self.d_model, -1).permute(0, 1, 3, 2)  # [30, 32, 704, d_model]
        q_flat = q_flat.reshape(s * n, -1, self.d_model).permute(1, 0, 2)  # [seq_len=704, batch_size=960, d_model]

        k_flat = k_proj.permute(2, 0, 1, 3, 4).reshape(s, n, self.d_model, -1).permute(0, 1, 3, 2)
        k_flat = k_flat.reshape(s * n, -1, self.d_model).permute(1, 0, 2)  # [704, 960, d_model]

        v_flat = v_proj.permute(2, 0, 1, 3, 4).reshape(s, n, self.d_model, -1).permute(0, 1, 3, 2)
        v_flat = v_flat.reshape(s * n, -1, self.d_model).permute(1, 0, 2)  # [704, 960, d_model]

        # 步骤4：交叉注意力计算（q来自Initial，k/v来自H3DA）
        attn_out, _ = self.multihead_attn(query=q_flat, key=k_flat, value=v_flat)  # [704, 960, d_model]

        # 步骤5：维度恢复为3D特征格式（匹配kv的空间时序维度）
        attn_out = attn_out.permute(1, 0, 2).reshape(n, s, -1, self.d_model)  # [32, 30, 704, d_model]
        attn_out = attn_out.permute(0, 3, 1, 2).reshape(n, self.d_model, s, 32, 22)  # [32, d_model, 30, 32, 22]

        # 步骤6：投影回H3DA通道数 + 残差融合（与原H3DA特征融合）
        attn_proj = self.out_proj(attn_out)  # [32, 128, 30, 32, 22]（通道匹配kv）
        fused = self.norm(kv + attn_proj)  # 残差连接（保留原特征+补充注意力特征）
        fused = self.relu(fused)

        return fused


class TemporalCrossAttention(nn.Module):
    """
    修改后：Temporal-Transformer输出通道为256
    - 输入：Spatial-Transformer输出（[32, 128, 30, 32, 22]）
    - 输出：[32, 256, 30, 32, 22]（通道数从128→256）
    """

    def __init__(self, in_ch=128, target_ch=256, d_model=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.target_ch = target_ch  # 新增：目标输出通道数（256）
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        # 1. 通道投影：Spatial的128维→d_model（q/k/v独立投影）
        self.q_proj = nn.Conv3d(in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.k_proj = nn.Conv3d(in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.v_proj = nn.Conv3d(in_ch, d_model, kernel_size=(1, 1, 1), stride=1, padding=0)

        # 2. 空间池化：压缩h/w→1，突出时序维度
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # 3. 多头注意力（自注意力，q/k/v来自Spatial输出）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False
        )

        # 4. 输出投影：d_model→256通道（核心修改点）
        self.out_proj = nn.Conv3d(d_model, self.target_ch, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.norm = nn.BatchNorm3d(self.target_ch)  # 归一化层通道数同步改为256
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, spatial_feat):
        # 输入：Spatial-Transformer输出 [32, 128, 30, 32, 22]
        n, _, s, h_kv, w_kv = spatial_feat.size()

        # 步骤1：q/k/v预处理（通道投影→空间池化）
        q_proj = self.q_proj(spatial_feat)  # [32, d_model, 30, 32, 22]
        q_pool = self.spatial_pool(q_proj)  # [32, d_model, 30, 1, 1]

        k_proj = self.k_proj(spatial_feat)  # [32, d_model, 30, 32, 22]
        k_pool = self.spatial_pool(k_proj)  # [32, d_model, 30, 1, 1]

        v_proj = self.v_proj(spatial_feat)  # [32, d_model, 30, 32, 22]
        v_pool = self.spatial_pool(v_proj)  # [32, d_model, 30, 1, 1]

        # 步骤2：维度调整为MultiheadAttention格式
        q_flat = q_pool.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # [30, 32, d_model]
        k_flat = k_pool.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # [30, 32, d_model]
        v_flat = v_pool.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # [30, 32, d_model]

        # 步骤3：注意力计算
        attn_out, _ = self.multihead_attn(query=q_flat, key=k_flat, value=v_flat)  # [30, 32, d_model]

        # 步骤4：维度恢复+投影到256通道
        attn_out = attn_out.permute(1, 2, 0).unsqueeze(-1).unsqueeze(-1)  # [32, d_model, 30, 1, 1]
        # 上采样恢复空间维度（32×22）
        attn_upsample = F.interpolate(
            attn_out, size=(s, h_kv, w_kv), mode='trilinear', align_corners=True
        )  # [32, d_model, 30, 32, 22]

        # 投影到256通道（核心修改：输出通道从128→256）
        attn_proj = self.out_proj(attn_upsample)  # [32, 256, 30, 32, 22]

        # 步骤5：残差融合（Temporal特征 + 原Spatial特征，需先将Spatial投影到256通道）
        # 新增：Spatial特征从128→256通道，确保残差连接维度匹配
        spatial_feat_256 = nn.Conv3d(128, self.target_ch, kernel_size=1, stride=1, padding=0).to(spatial_feat.device)(
            spatial_feat)
        fused = self.norm(spatial_feat_256 + attn_proj)  # 残差连接，保留Spatial信息
        fused = self.relu(fused)

        return fused  # 输出：[32, 256, 30, 32, 22]