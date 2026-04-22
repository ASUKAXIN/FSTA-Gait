import copy
import os
import inspect
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple

# 定义一个 NoOp 类，用于在某些情况下提供无操作的方法
class NoOp:
    def __getattr__(self, *args):
        """
        当尝试访问 NoOp 类的属性时，返回一个无操作的函数
        :param args: 未使用的参数
        :return: 无操作的函数
        """
        def no_op(*args, **kwargs): pass
        return no_op

# 定义一个 Odict 类，继承自 OrderedDict，用于处理字典数据
class Odict(OrderedDict):
    def append(self, odict):
        """
        将另一个有序字典的内容追加到当前有序字典中
        :param odict: 要追加的有序字典
        """
        dst_keys = self.keys()
        for k, v in odict.items():
            # 如果值不是列表，则将其转换为列表
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                # 如果键已存在于当前字典中，根据值的类型进行追加
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                # 如果键不存在，则直接添加到当前字典中
                self[k] = v

# 定义一个函数，用于创建一个命名元组
def Ntuple(description, keys, values):
    """
    创建一个命名元组
    :param description: 命名元组的描述
    :param keys: 命名元组的键，可以是单个键或键的列表
    :param values: 命名元组的值，可以是单个值或值的列表
    :return: 命名元组实例
    """
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)

# 定义一个函数，用于获取对象所需的有效参数
def get_valid_args(obj, input_args, free_keys=[]):
    """
    获取对象所需的有效参数
    :param obj: 函数或类对象
    :param input_args: 输入的参数字典
    :param free_keys: 允许的自由参数列表
    :return: 有效参数的字典
    """
    if inspect.isfunction(obj):
        # 如果 obj 是函数，获取函数的参数列表
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        # 如果 obj 是类，获取类的 __init__ 方法的参数列表
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            # 如果键在预期的参数列表中，则添加到有效参数字典中
            expected_args[k] = v
        elif k in free_keys:
            # 如果键在自由参数列表中，则忽略
            pass
        else:
            # 如果键既不在预期参数列表中，也不在自由参数列表中，则记录为意外参数
            unexpect_keys.append(k)
    if unexpect_keys != []:
        # 如果存在意外参数，记录日志
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args

# 定义一个函数，用于从多个源对象中获取指定名称的属性
def get_attr_from(sources, name):
    """
    从多个源对象中获取指定名称的属性
    :param sources: 源对象的列表
    :param name: 要获取的属性名称
    :return: 获取到的属性值
    """
    try:
        return getattr(sources[0], name)
    except:
        # 如果第一个源对象中没有该属性，则递归地从剩余的源对象中查找
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)

# 定义一个函数，用于判断对象是否为列表或元组
def is_list_or_tuple(x):
    """
    判断对象是否为列表或元组
    :param x: 要判断的对象
    :return: 如果是列表或元组，返回 True；否则返回 False
    """
    return isinstance(x, (list, tuple))

# 定义一个函数，用于判断对象是否为布尔类型
def is_bool(x):
    """
    判断对象是否为布尔类型
    :param x: 要判断的对象
    :return: 如果是布尔类型，返回 True；否则返回 False
    """
    return isinstance(x, bool)

# 定义一个函数，用于判断对象是否为字符串类型
def is_str(x):
    """
    判断对象是否为字符串类型
    :param x: 要判断的对象
    :return: 如果是字符串类型，返回 True；否则返回 False
    """
    return isinstance(x, str)

# 定义一个函数，用于判断对象是否为列表或 nn.ModuleList 类型
def is_list(x):
    """
    判断对象是否为列表或 nn.ModuleList 类型
    :param x: 要判断的对象
    :return: 如果是列表或 nn.ModuleList 类型，返回 True；否则返回 False
    """
    return isinstance(x, list) or isinstance(x, nn.ModuleList)

# 定义一个函数，用于判断对象是否为字典、OrderedDict 或 Odict 类型
def is_dict(x):
    """
    判断对象是否为字典、OrderedDict 或 Odict 类型
    :param x: 要判断的对象
    :return: 如果是字典、OrderedDict 或 Odict 类型，返回 True；否则返回 False
    """
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)

# 定义一个函数，用于判断对象是否为 torch.Tensor 类型
def is_tensor(x):
    """
    判断对象是否为 torch.Tensor 类型
    :param x: 要判断的对象
    :return: 如果是 torch.Tensor 类型，返回 True；否则返回 False
    """
    return isinstance(x, torch.Tensor)

# 定义一个函数，用于判断对象是否为 numpy.ndarray 类型
def is_array(x):
    """
    判断对象是否为 numpy.ndarray 类型
    :param x: 要判断的对象
    :return: 如果是 numpy.ndarray 类型，返回 True；否则返回 False
    """
    return isinstance(x, np.ndarray)

# 定义一个函数，用于将 torch.Tensor 转换为 numpy.ndarray
def ts2np(x):
    """
    将 torch.Tensor 转换为 numpy.ndarray
    :param x: 要转换的 torch.Tensor
    :return: 转换后的 numpy.ndarray
    """
    return x.cpu().data.numpy()

# 定义一个函数，用于将 torch.Tensor 转换为 autograd.Variable 并移动到 GPU 上
def ts2var(x, **kwargs):
    """
    将 torch.Tensor 转换为 autograd.Variable 并移动到 GPU 上
    :param x: 要转换的 torch.Tensor
    :param kwargs: 传递给 autograd.Variable 的额外参数
    :return: 转换后的 autograd.Variable
    """
    return autograd.Variable(x, **kwargs).cuda()

# 定义一个函数，用于将 numpy.ndarray 转换为 autograd.Variable 并移动到 GPU 上
def np2var(x, **kwargs):
    """
    将 numpy.ndarray 转换为 autograd.Variable 并移动到 GPU 上
    :param x: 要转换的 numpy.ndarray
    :param kwargs: 传递给 autograd.Variable 的额外参数
    :return: 转换后的 autograd.Variable
    """
    return ts2var(torch.from_numpy(x), **kwargs)

# 定义一个函数，用于将列表转换为 autograd.Variable 并移动到 GPU 上
def list2var(x, **kwargs):
    """
    将列表转换为 autograd.Variable 并移动到 GPU 上
    :param x: 要转换的列表
    :param kwargs: 传递给 autograd.Variable 的额外参数
    :return: 转换后的 autograd.Variable
    """
    return np2var(np.array(x), **kwargs)

# 定义一个函数，用于创建目录
def mkdir(path):
    """
    创建目录，如果目录不存在
    :param path: 要创建的目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)

# 定义一个函数，用于合并两个配置字典
def MergeCfgsDict(src, dst):
    """
    合并两个配置字典
    :param src: 源配置字典
    :param dst: 目标配置字典
    """
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            # 如果键不在目标字典中，或者值不是字典类型，则直接添加到目标字典中
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                # 如果键对应的值都是字典类型，则递归合并
                MergeCfgsDict(src[k], dst[k])
            else:
                # 否则，直接更新目标字典中的值
                dst[k] = v

# 定义一个函数，用于克隆模块 N 次
def clones(module, N):
    """
    克隆模块 N 次
    :param module: 要克隆的模块
    :param N: 克隆的次数
    :return: 包含 N 个克隆模块的 nn.ModuleList
    """
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 定义一个函数，用于加载配置文件
def config_loader(path):
    """
    加载配置文件，合并自定义配置和默认配置
    :param path: 自定义配置文件的路径
    :return: 合并后的配置字典
    """
    with open(path, 'r', encoding='utf-8') as stream:
        # 读取自定义配置文件
        src_cfgs = yaml.safe_load(stream)
    with open("./configs/default.yaml", 'r', encoding='utf-8') as stream:
        # 读取默认配置文件
        dst_cfgs = yaml.safe_load(stream)
    # 合并自定义配置和默认配置
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs

# 定义一个函数，用于初始化随机种子
def init_seeds(seed=0, cuda_deterministic=True):
    """
    初始化随机种子
    :param seed: 随机种子的值，默认为 0
    :param cuda_deterministic: 是否使用确定性的 CUDA 算法，默认为 True
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        # 使用确定性的 CUDA 算法，速度较慢，但结果更可重复
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        # 使用非确定性的 CUDA 算法，速度较快，但结果不太可重复
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# 定义一个信号处理函数，用于处理 Ctrl+C 或 Ctrl+Z 信号
def handler(signum, frame):
    """
    处理 Ctrl+C 或 Ctrl+Z 信号
    :param signum: 信号编号
    :param frame: 当前栈帧
    """
    logging.info('Ctrl+c/z pressed')
    # 杀死所有运行 main.py 的进程
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')

# 定义一个函数，用于在分布式训练中收集所有进程的特征
# def ddp_all_gather(features, dim=0, requires_grad=True):
#     '''
#         inputs: [n, ...]
#     '''
#     """
#     在分布式训练中收集所有进程的特征
#     :param features: 要收集的特征
#     :param dim: 拼接特征的维度，默认为 0
#     :param requires_grad: 是否需要梯度，默认为 True
#     :return: 收集并拼接后的特征
#     """
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     feature_list = [torch.ones_like(features) for _ in range(world_size)]
#     # 收集所有进程的特征
#     torch.distributed.all_gather(feature_list, features.contiguous())
#
#     if requires_grad:
#         # 如果需要梯度，将当前进程的特征替换为原始特征
#         feature_list[rank] = features
#     # 拼接所有进程的特征
#     feature = torch.cat(feature_list, dim=dim)
#     return feature

def ddp_all_gather(tensor, requires_grad=True):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            return tensor
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, tensor)
        output = torch.cat(tensor_list, dim=0)
        return output
    else:
        return tensor


# 定义一个 DDPPassthrough 类，继承自 DDP，用于处理属性访问
# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        """
        处理属性访问，优先从父类获取属性，如果不存在则从模块中获取
        :param name: 要访问的属性名称
        :return: 获取到的属性值
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# 定义一个函数，用于将模块包装为分布式数据并行模块
def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    """
    将模块包装为分布式数据并行模块
    :param module: 要包装的模块
    :param find_unused_parameters: 是否查找未使用的参数，默认为 False
    :param kwargs: 传递给 DDPPassthrough 的额外参数
    :return: 包装后的分布式数据并行模块
    """
    if len(list(module.parameters())) == 0:
        # 如果模块没有参数，则直接返回模块
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    # 将模块包装为 DDPPassthrough 模块
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module

# 定义一个函数，用于计算网络的参数数量
def params_count(net):
    """
    计算网络的参数数量
    :param net: 要计算参数数量的网络
    :return: 格式化后的参数数量字符串
    """
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)