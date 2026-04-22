import time
import torch

import numpy as np
import torchvision.utils as vutils
import os.path as osp
from time import strftime, localtime

from torch.utils.tensorboard import SummaryWriter
from .common import is_list, is_tensor, ts2np, mkdir, Odict, NoOp
import logging

# 定义消息管理类，用于管理训练过程中的信息记录和输出
class MessageManager:
    def __init__(self):
        # 初始化一个有序字典，用于存储各种信息
        self.info_dict = Odict()
        # 定义可以写入 TensorBoard 的数据类型
        self.writer_hparams = ['image', 'scalar']
        # 记录当前时间，用于计算时间间隔
        self.time = time.time()
        self.logger = NoOp()
        self.iteration = 0
        self.log_iter = 100  # 默认日志迭代间隔为100
        self.writer = NoOp()

    # 初始化消息管理器
    def init_manager(self, save_path, log_to_file, log_iter, iteration):
        # 当前迭代次数
        self.iteration = iteration
        # 日志记录的迭代间隔
        self.log_iter = log_iter
        # 创建存储 TensorBoard 摘要的文件夹
        mkdir(osp.join(save_path, "summary/"))
        # 初始化 TensorBoard 的 SummaryWriter，指定存储路径和起始迭代步骤
        self.writer = SummaryWriter(
            osp.join(save_path, "summary/"), purge_step=self.iteration)
        # 初始化日志记录器
        self.init_logger(save_path, log_to_file)

    # 初始化日志记录器
    def init_logger(self, save_path, log_to_file):
        # 初始化名为 'opengait' 的日志记录器
        self.logger = logging.getLogger('opengait')
        # 设置日志记录级别为 INFO
        self.logger.setLevel(logging.INFO)
        # 避免日志信息向上传播
        self.logger.propagate = False
        # 定义日志信息的格式
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_file:
            # 创建存储日志文件的文件夹
            mkdir(osp.join(save_path, "logs/"))
            # 定义日志文件的名称，使用当前时间作为文件名
            vlog = logging.FileHandler(
                osp.join(save_path, "logs/", strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'))
            # 设置文件日志记录级别为 INFO
            vlog.setLevel(logging.INFO)
            # 为文件日志记录器设置日志格式
            vlog.setFormatter(formatter)
            # 将文件日志记录器添加到日志记录器中
            self.logger.addHandler(vlog)

        # 定义控制台日志记录器
        console = logging.StreamHandler()
        # 为控制台日志记录器设置日志格式
        console.setFormatter(formatter)
        # 设置控制台日志记录级别为 DEBUG
        console.setLevel(logging.DEBUG)
        # 将控制台日志记录器添加到日志记录器中
        self.logger.addHandler(console)

    # 向 info_dict 中追加信息
    def append(self, info):
        for k, v in info.items():
            # 如果 v 不是列表，则将其转换为列表
            v = [v] if not is_list(v) else v
            # 如果 v 是张量，则将其转换为 numpy 数组
            v = [ts2np(_) if is_tensor(_) else _ for _ in v]
            # 更新 info_dict 中的信息
            info[k] = v
        # 将更新后的信息追加到 info_dict 中
        self.info_dict.append(info)

    # 清空 info_dict 并刷新 TensorBoard 的写入器
    def flush(self):
        # 清空 info_dict
        self.info_dict.clear()
        # 刷新 TensorBoard 的写入器
        self.writer.flush()

    # 将摘要信息写入 TensorBoard
    def write_to_tensorboard(self, summary):
        for k, v in summary.items():
            # 获取摘要信息的模块名称
            module_name = k.split('/')[0]
            if module_name not in self.writer_hparams:
                # 如果模块名称不在允许的写入类型中，则记录警告信息
                self.log_warning(
                    'Not Expected --Summary-- type [{}] appear!!!{}'.format(k, self.writer_hparams))
                continue
            # 获取写入 TensorBoard 的名称
            board_name = k.replace(module_name + "/", '')
            # 获取 TensorBoard 写入器的相应方法
            writer_module = getattr(self.writer, 'add_' + module_name)
            # 如果 v 是张量，则将其分离
            v = v.detach() if is_tensor(v) else v
            # 如果是图像类型，则对图像进行归一化处理
            v = vutils.make_grid(
                v, normalize=True, scale_each=True) if 'image' in module_name else v
            if module_name == 'scalar':
                try:
                    # 如果是标量类型，则计算其均值
                    v = v.mean()
                except:
                    v = v
            # 调用相应的写入方法将信息写入 TensorBoard
            writer_module(board_name, v, self.iteration)

    # 记录训练信息
    def log_training_info(self):
        # 获取当前时间
        now = time.time()
        # 构建训练信息字符串，包含当前迭代次数和本次迭代的耗时
        string = "Iteration {:0>5}, Cost {:.2f}s".format(
            self.iteration, now-self.time)
        for i, (k, v) in enumerate(self.info_dict.items()):
            if 'scalar' not in k:
                continue
            # 处理标量信息的键名
            k = k.replace('scalar/', '').replace('/', '_')
            # 根据是否是最后一个信息决定是否换行
            end = "\n" if i == len(self.info_dict)-1 else ""
            # 向训练信息字符串中添加标量信息
            string += ", {0}={1:.4f}".format(k, np.mean(v), end=end)
        # 记录训练信息
        self.log_info(string)
        print("Loss information: " + string)
        # 重置时间记录
        self.reset_time()

    # 重置时间记录
    def reset_time(self):
        # 更新时间记录为当前时间
        self.time = time.time()

    # 训练步骤，处理训练信息和摘要信息
    def train_step(self, info, summary):
        # 迭代次数加 1
        self.iteration += 1
        # 向 info_dict 中追加训练信息
        self.append(info)
        if self.iteration % self.log_iter == 0:
            # 如果达到日志记录的迭代间隔，则记录训练信息
            self.log_training_info()

            # 清空 info_dict 并刷新 TensorBoard 的写入器
            self.flush()
            # 将摘要信息写入 TensorBoard
            self.write_to_tensorboard(summary)

    # 记录调试信息
    def log_debug(self, *args, **kwargs):
        # 调用日志记录器的 debug 方法记录调试信息
        self.logger.debug(*args, **kwargs)

    # 记录普通信息
    def log_info(self, *args, **kwargs):
        # 调用日志记录器的 info 方法记录普通信息
        self.logger.info(*args, **kwargs)

    # 记录警告信息
    def log_warning(self, *args, **kwargs):
        # 调用日志记录器的 warning 方法记录警告信息
        self.logger.warning(*args, **kwargs)

# 实例化消息管理器
msg_mgr = MessageManager()
# 定义一个空操作类的实例
noop = NoOp()

# 获取消息管理器实例
def get_msg_mgr():
    # 不使用分布式训练，直接返回消息管理器实例
    return msg_mgr