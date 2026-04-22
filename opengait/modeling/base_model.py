"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod
from torch.utils.data import RandomSampler

from . import backbones
from .loss_aggregator import LossAggregator

from opengait.utils.msg_manager import MessageManager
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from utils import NoOp
from opengait.utils.msg_manager import get_msg_mgr

__all__ = ['BaseModel']



class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader.

        Args:
            data_cfg (dict): 数据配置字典，包含数据集的相关配置信息。
        """
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here.

        Args:
            model_cfg (dict): 模型配置字典，包含模型结构的相关配置信息。
        """
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer.

        Args:
            optimizer_cfg (dict): 优化器配置字典，包含优化器的相关配置信息，如学习率、动量等。
        """
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler.

        Args:
            scheduler_cfg (dict): 学习率调度器配置字典，包含调度器的相关配置信息，如衰减率、衰减步数等。
        """
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler.

        Args:
            iteration (int): 当前训练的迭代次数。
        """
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler.

        Args:
            restore_hint (int or str): 恢复检查点的提示信息，可能是迭代次数或检查点文件路径。
        """
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting.

        Args:
            inputs (tuple or list): 输入数据，可能包含图像、标签等。
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step.

        Args:
            loss_num (int): 损失函数的数量。

        Returns:
            bool: 训练步骤是否成功完成。
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule.

        Args:
            model (BaseModel): 要训练的模型实例。
        """
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule.

        Args:
            model (BaseModel): 要测试的模型实例。
        """
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager. 消息管理器，用于记录和输出信息。
        cfgs: the configs. 所有的配置信息。
        iteration: the current iteration of the model. 模型当前的迭代次数。
        engine_cfg: the configs of the engine(train or test). 训练或测试引擎的配置信息。
        save_path: the path to save the checkpoints. 保存检查点的路径。
    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
            cfgs (dict): All of the configs. 所有的配置信息。
            training (bool): Whether the model is in training mode. 模型是否处于训练模式。
        """

        super(BaseModel, self).__init__()
        # 获取消息管理器实例
        self.msg_mgr = NoOp()
        # 保存所有配置信息
        self.cfgs = cfgs
        # 初始化当前迭代次数为 0
        self.iteration = 0
        # 根据训练或测试模式选择相应的引擎配置
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            # 如果引擎配置为空，抛出异常
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            # 如果处于训练模式且启用了混合精度训练，初始化梯度缩放器
            self.Scaler = GradScaler()
        # 构建保存检查点的路径
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        # 构建网络
        self.build_network(cfgs['model_cfg'])
        # 初始化网络参数
        self.init_parameters()
        # 获取训练数据的变换操作
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])

        # 记录数据配置信息
        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            # 如果处于训练模式，获取训练数据加载器
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            # 如果处于测试模式或训练时需要测试，获取测试数据加载器
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            # 获取测试数据的变换操作
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])

        # 设置当前使用的 CUDA 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 将模型移动到指定的 CUDA 设备上
        self.to(device=self.device)

        if training:
            # 如果处于训练模式，初始化损失聚合器
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            # 获取优化器
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            # 获取学习率调度器
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        # 设置模型的训练或评估模式
        self.train(training)
        # 获取恢复检查点的提示信息
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            # 如果提示信息不为 0，恢复模型的检查点
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model.

        Args:
            backbone_cfg (dict or list): 骨干网络的配置信息，可以是字典或字典列表。

        Returns:
            nn.Module: 骨干网络的实例。
        """
        if is_dict(backbone_cfg):
            # 如果配置信息是字典，根据配置信息获取骨干网络类
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            # 获取有效的参数
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            # 实例化骨干网络
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            # 如果配置信息是列表，递归调用 get_backbone 方法获取每个骨干网络实例，并组成模块列表
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        # 如果配置信息既不是字典也不是列表，抛出值错误异常
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        """Build the network based on the model configuration.

        Args:
            model_cfg (dict): 模型配置字典，包含模型结构的相关配置信息。
        """
        if 'backbone_cfg' in model_cfg.keys():
            # 如果模型配置中包含骨干网络配置，获取骨干网络实例并保存到模型中
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        """Initialize the parameters of the network."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                # 如果是卷积层，使用 Xavier 均匀分布初始化权重
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    # 如果卷积层有偏置，将偏置初始化为 0
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                # 如果是全连接层，使用 Xavier 均匀分布初始化权重
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    # 如果全连接层有偏置，将偏置初始化为 0
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    # 如果是批量归一化层且启用了仿射变换，使用正态分布初始化权重，偏置初始化为 0
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True):
        """Get the data loader based on the data configuration.

        Args:
            data_cfg (dict): 数据配置字典，包含数据集的相关配置信息。
            train (bool): 是否为训练数据加载器。

        Returns:
            torch.utils.data.DataLoader: 数据加载器实例。
        """
        # 根据训练或测试模式选择相应的采样器配置
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        # 实例化数据集
        dataset = DataSet(data_cfg, train)

        # 根据采样器配置获取采样器类
        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        # 获取有效的采样器参数
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        # 实例化采样器
        sampler = Sampler(dataset, **vaild_args)

        # 创建数据加载器
        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader

    def get_optimizer(self, optimizer_cfg):
        """Get the optimizer based on the optimizer configuration.

        Args:
            optimizer_cfg (dict): 优化器配置字典，包含优化器的相关配置信息，如学习率、动量等。

        Returns:
            torch.optim.Optimizer: 优化器实例。
        """
        # 记录优化器配置信息
        self.msg_mgr.log_info(optimizer_cfg)
        # 根据优化器配置获取优化器类
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        # 获取有效的优化器参数
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        # 实例化优化器，只对需要梯度更新的参数进行优化
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        """Get the scheduler based on the scheduler configuration.

        Args:
            scheduler_cfg (dict): 学习率调度器配置字典，包含调度器的相关配置信息，如衰减率、衰减步数等。

        Returns:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器实例。
        """
        # 记录学习率调度器配置信息
        self.msg_mgr.log_info(scheduler_cfg)
        # 根据调度器配置获取调度器类
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        # 获取有效的调度器参数
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        # 实例化调度器
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler.

        Args:
            iteration (int): 当前训练的迭代次数。
        """
        # if torch.distributed.get_rank() == 0:
        # 如果是主进程，创建保存检查点的文件夹
        print(f"Save")
        mkdir(osp.join(self.save_path, "checkpoints/"))
        # 获取保存的名称
        save_name = self.engine_cfg['save_name']
        # 构建检查点字典，包含模型参数、优化器状态、调度器状态和迭代次数
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': iteration}
        # 保存检查点文件
        torch.save(checkpoint,
                    osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        """Load the checkpoint from the given file.

        Args:
            save_name (str): 检查点文件的路径。
        """
        # 获取是否严格加载检查点的配置
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        # 加载检查点文件
        # checkpoint = torch.load(save_name, map_location=torch.device(
        #     "cuda", self.device))

        # 修改后的代码
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(save_name, map_location=device)

        # 获取模型的状态字典
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            # 如果不严格加载，记录恢复的参数列表
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        # 加载模型的状态字典
        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                # 如果不重置优化器且检查点中包含优化器状态，加载优化器状态
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                # 否则，记录警告信息
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                # 如果不重置调度器且检查点中包含调度器状态，加载调度器状态
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                # 如果不重置调度器且检查点中包含调度器状态，没有成功加载调度器状态，则记录警告信息
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
            # 记录信息，表示从指定路径成功恢复参数
            self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)
    def resume_ckpt(self, restore_hint):
        """
        从检查点恢复模型，包括模型参数、优化器和调度器。

        Args:
            restore_hint (int or str): 恢复提示信息。如果是整数，则表示迭代次数；如果是字符串，则表示检查点文件的路径。
        """
        # 如果恢复提示信息是整数
        if isinstance(restore_hint, int):
            # 获取保存的名称
            save_name = self.engine_cfg['save_name']
            # 构建检查点文件的完整路径
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            # 更新当前迭代次数为恢复提示信息指定的迭代次数
            self.iteration = restore_hint
        # 如果恢复提示信息是字符串
        elif isinstance(restore_hint, str):
            # 直接将恢复提示信息作为检查点文件的路径
            save_name = restore_hint
            # 将当前迭代次数重置为 0
            self.iteration = 0
        # 如果恢复提示信息既不是整数也不是字符串
        else:
            # 抛出值错误异常，提示恢复提示信息的类型不支持
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        # 调用 _load_ckpt 方法加载检查点文件
        self._load_ckpt(save_name)

    def fix_BN(self):
        """
        固定模型中所有 BatchNorm 层的参数，将其设置为评估模式。
        """
        # 遍历模型的所有模块
        for module in self.modules():
            # 获取当前模块的类名
            classname = module.__class__.__name__
            # 如果类名中包含 'BatchNorm'
            if classname.find('BatchNorm') != -1:
                # 将该模块设置为评估模式
                module.eval()

    def inputs_pretreament(self, inputs):
        """
        对输入数据进行预处理，包括数据转换和格式转换。

        Args:
            inputs (tuple): 输入数据，包含序列数据、标签、类型、视角和序列长度。

        Returns:
            tuple: 预处理后的训练数据，包括输入、标签和一些元数据。
        """
        # 解包输入数据
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        # 根据模型的训练状态选择使用训练数据的变换操作还是测试数据的变换操作
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        # 检查输入数据的类型数量和变换操作的数量是否一致
        if len(seqs_batch) != len(seq_trfs):
            # 如果不一致，抛出值错误异常，提示数量不匹配
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        # 根据模型的训练状态确定是否需要计算梯度
        requires_grad = bool(self.training)
        # 对每个序列数据应用相应的变换操作，并转换为 PyTorch 张量
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        # 保留类型数据
        typs = typs_batch
        # 保留视角数据
        vies = vies_batch

        # 将标签数据转换为 PyTorch 张量
        labs = list2var(labs_batch).long()

        # 如果序列长度数据不为空
        if seqL_batch is not None:
            # 将序列长度数据转换为 PyTorch 张量
            seqL_batch = np2var(seqL_batch).int()
        # 保存序列长度数据
        seqL = seqL_batch

        # 如果序列长度数据不为空
        if seqL is not None:
            # 计算序列长度的总和
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            # 根据序列长度总和截取输入数据
            ipts = [_[:, :seqL_sum] for _ in seqs]
        # 如果序列长度数据为空
        else:
            # 直接将处理后的序列数据作为输入数据
            ipts = seqs
        # 删除不再使用的序列数据，释放内存
        del seqs
        # 返回预处理后的输入数据、标签、类型、视角和序列长度
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """
        执行一次训练步骤，包括损失反向传播、优化器更新和学习率调度器更新。

        Args:
            loss_sum (torch.Tensor): 当前批次的损失值。

        Returns:
            bool: 如果训练步骤成功完成，返回 True；如果由于梯度缩放问题跳过训练步骤，返回 False。
        """
        # 清空优化器中的梯度信息
        self.optimizer.zero_grad()
        self.msg_mgr = get_msg_mgr()
        # 如果损失值小于等于 1e-9
        if loss_sum <= 1e-9:
            # 记录警告信息，提示损失值过小，但训练过程将继续
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        # 如果启用了混合精度训练
        if self.engine_cfg['enable_float16']:
            # 使用梯度缩放器对损失值进行缩放，并进行反向传播
            self.Scaler.scale(loss_sum).backward()
            # 使用梯度缩放器更新优化器
            self.Scaler.step(self.optimizer)
            # 获取当前的梯度缩放比例
            scale = self.Scaler.get_scale()
            # 更新梯度缩放器
            self.Scaler.update()
            # 如果缩放比例发生变化，说明在更新过程中出现了 NaN 或 Inf，跳过本次训练步骤
            if scale != self.Scaler.get_scale():
                # 记录调试信息，提示训练步骤跳过
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                # 返回 False 表示训练步骤未成功完成
                return False
        # 如果未启用混合精度训练
        else:
            # 直接进行损失反向传播
            loss_sum.backward()
            # 更新优化器
            self.optimizer.step()

        # 迭代次数加 1
        self.iteration += 1
        # 更新学习率调度器
        self.scheduler.step()
        # 返回 True 表示训练步骤成功完成
        return True

    def inference(self):
        """
        对所有测试数据进行推理，计算特征。

        Args:
            rank (int): 当前进程的排名。

        Returns:
            Odict: 包含推理结果的有序字典。
        """
        # 获取测试数据加载器的总批次数量
        total_size = len(self.test_loader)
        # 如果当前进程的排名为 0
        # if rank == 0:
        #     # 使用 tqdm 显示进度条，描述为 'Transforming'
        #     pbar = tqdm(total=total_size, desc='Transforming')
        # # 如果当前进程的排名不为 0
        # else:
        #     # 使用 NoOp 类，不显示进度条
        #     pbar = NoOp()

        pbar = tqdm(total=total_size, desc='Transforming')
        # 获取测试数据加载器的批次大小
        batch_size = self.test_loader.batch_sampler.batch_size
        # 剩余未处理的批次数量
        rest_size = total_size
        # 初始化一个有序字典，用于存储推理结果
        info_dict = Odict()
        # 遍历测试数据加载器中的每个批次
        for inputs in self.test_loader:
            # 对输入数据进行预处理
            ipts = self.inputs_pretreament(inputs)
            # 使用自动混合精度进行推理
            with autocast(enabled=self.engine_cfg['enable_float16']):
                # 前向传播，获取推理结果
                retval = self.forward(ipts)
                # 提取推理特征
                inference_feat = retval['inference_feat']
                # 对推理特征进行分布式收集
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                # 删除不再使用的推理结果
                del retval
            # 将推理特征转换为 NumPy 数组
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            # 将推理特征添加到有序字典中
            info_dict.append(inference_feat)
            # 剩余未处理的批次数量减去当前批次大小
            rest_size -= batch_size
            # 如果剩余批次数量大于等于 0
            if rest_size >= 0:
                # 更新进度条的大小为当前批次大小
                update_size = batch_size
            # 如果剩余批次数量小于 0
            else:
                # 更新进度条的大小为总批次数量对批次大小取余的结果
                update_size = total_size % batch_size
            # 更新进度条
            pbar.update(update_size)
        # 关闭进度条
        pbar.close()
        # 对有序字典中的每个键值对进行处理
        for k, v in info_dict.items():
            # 将所有批次的推理特征拼接成一个数组，并截取前 total_size 个元素
            v = np.concatenate(v)[:total_size]
            # 更新有序字典中的值
            info_dict[k] = v
        # 返回包含推理结果的有序字典
        return info_dict

    @staticmethod
    def run_train(model):
        """
        接受模型实例对象，运行整个训练循环。

        Args:
            model (BaseModel): 要训练的模型实例。
        """
        # 获取消息管理器实例
        model.msg_mgr = get_msg_mgr()
        # 遍历训练数据加载器中的每个批次
        for inputs in model.train_loader:
            # 对输入数据进行预处理
            ipts = model.inputs_pretreament(inputs)
            # 使用自动混合精度进行前向传播
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # 前向传播，获取训练特征和可视化摘要
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                # 删除不再使用的推理结果
                del retval
            # 计算损失总和和损失信息
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            # 执行一次训练步骤
            ok = model.train_step(loss_sum)
            # print(f"Current iteration: {model.iteration}")
            # 如果训练步骤未成功完成
            if not ok:
                # 跳过本次循环，继续下一个批次的训练
                continue

            # 将损失信息更新到可视化摘要中
            visual_summary.update(loss_info)
            # 将当前学习率添加到可视化摘要中
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            # 记录训练步骤的损失信息和可视化摘要
            model.msg_mgr.train_step(loss_info, visual_summary)
            # model.msg_mgr.log_info(loss_info)

            if model.iteration % 1000 == 0:
                print(f"Current iteration: {model.iteration}")
                print('loss_sum={0:.8f}'.format(loss_sum))

            # 如果当前迭代次数是保存间隔的整数倍
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # 保存当前迭代次数的检查点
                model.save_ckpt(model.iteration)

                # 如果配置中设置了在训练过程中进行测试
                if model.engine_cfg['with_test']:
                    # 记录信息，表示开始运行测试
                    model.msg_mgr.log_info("Running test...")
                    # 将模型设置为评估模式
                    model.eval()
                    # 运行测试，获取测试结果
                    result_dict = BaseModel.run_test(model)
                    # 将模型设置为训练模式
                    model.train()
                    # 如果配置中设置了固定 BatchNorm 层
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        # 固定 BatchNorm 层的参数
                        model.fix_BN()
                    # 如果测试结果不为空
                    if result_dict:
                        # 将测试结果写入 TensorBoard
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    # 重置时间记录
                    model.msg_mgr.reset_time()
            # 如果当前迭代次数达到总迭代次数
            if model.iteration >= model.engine_cfg['total_iter']:
                # 跳出训练循环
                break

    @staticmethod
    def run_test(model):
        """
        接受模型实例对象，运行整个测试循环。

        Args:
            model (BaseModel): 要测试的模型实例。
        """
        # 获取评估器的配置信息
        evaluator_cfg = model.cfgs['evaluator_cfg']
        # 检查测试模式下的批次大小是否等于 GPU 数量
        # if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
        #     # 如果不相等，抛出值错误异常，提示批次大小和 GPU 数量不匹配
        #     raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
        #         evaluator_cfg['sampler']['batch_size'], torch.distributed.get_world_size()))
        # 获取当前进程的排名
        # rank = torch.distributed.get_rank()
        # # 在不计算梯度的情况下进行推理
        # with torch.no_grad():
        #     # 对测试数据进行推理，获取推理结果
        #     info_dict = model.inference(rank)
        # # 如果当前进程的排名为 0
        # if rank == 0:
        #     # 获取测试数据加载器
        loader = model.test_loader
        # 获取测试数据的标签列表
        label_list = loader.dataset.label_list
        # 获取测试数据的类型列表
        types_list = loader.dataset.types_list
        # 获取测试数据的视角列表
        views_list = loader.dataset.views_list

        info_dict = model.inference()
        # 将标签、类型和视角信息添加到推理结果中
        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        # 如果评估器配置中包含评估函数
        if 'eval_func' in evaluator_cfg.keys():
            # 获取评估函数的名称
            eval_func = evaluator_cfg["eval_func"]
        # 如果评估器配置中不包含评估函数
        else:
            # 默认使用 'identification' 作为评估函数
            eval_func = 'identification'
        # 从评估函数模块中获取评估函数
        eval_func = getattr(eval_functions, eval_func)
        # 获取评估函数的有效参数
        valid_args = get_valid_args(
            eval_func, evaluator_cfg, ['metric'])
        try:
            # 尝试获取测试数据集的名称
            dataset_name = model.cfgs['data_cfg']['test_dataset_name']
        except:
            # 如果获取失败，使用数据集的名称
            dataset_name = model.cfgs['data_cfg']['dataset_name']
        # 调用评估函数，对推理结果进行评估，并返回评估结果
        return eval_func(info_dict, dataset_name, **valid_args)