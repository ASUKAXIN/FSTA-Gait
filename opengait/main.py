import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
from utils.common import is_list, is_tensor, ts2np, mkdir, Odict, NoOp

# 创建命令行参数解析器，用于解析用户输入的命令行参数
parser = argparse.ArgumentParser(description='Main program for opengait.')
# 定义一个参数 --local_rank，用于指定当前进程在分布式训练中的本地排名，默认值为 0
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
# 定义一个参数 --local-rank，同样用于指定当前进程在分布式训练中的本地排名，适用于 PyTorch >= 2.0，默认值为 0
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
# 定义一个参数 --cfgs，用于指定配置文件的路径，默认值为 'config/default.yaml'
parser.add_argument('--cfgs', type=str,
                    default='your_path', help="path of config file")
# 定义一个参数 --phase，用于指定程序的运行阶段，可以是 'train' 或 'test'，默认值为 'train'
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
# 定义一个参数 --log_to_file，是一个布尔类型的标志，用于指定是否将日志输出到文件中
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
# 定义一个参数 --iter，用于指定要恢复的迭代次数，默认值为 0
parser.add_argument('--iter', default=0, help="iter to restore")
# 解析命令行参数
opt = parser.parse_args()


def initialization(cfgs, training):
    """
    初始化函数，用于初始化消息管理器、输出路径和随机种子
    :param cfgs: 配置文件字典
    :param training: 布尔值，表示当前是否处于训练阶段
    """
    # 获取消息管理器实例
    msg_mgr = get_msg_mgr()
    # 根据训练或测试阶段选择对应的配置
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    # 构建输出路径，格式为 output/<dataset>/<model>/<save_name>
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        # 如果是训练阶段，初始化消息管理器，设置日志输出路径、日志迭代间隔和恢复提示
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        # 如果是测试阶段，初始化日志记录器，设置日志输出路径
        msg_mgr.init_logger(output_path, opt.log_to_file)
    # 记录当前阶段的配置信息
    msg_mgr.log_info(engine_cfg)
    # 获取当前进程的排名作为随机种子
    # seed = torch.distributed.get_rank()

    # 初始化随机种子
    init_seeds(0)



def run_model(cfgs, training):
    """
    运行模型的函数，用于初始化模型、进行同步批归一化、固定批归一化层、包装模型为分布式数据并行模型，并执行训练或测试操作
    :param cfgs: 配置文件字典
    :param training: 布尔值，表示当前是否处于训练阶段
    """
    # 获取消息管理器实例
    msg_mgr = get_msg_mgr()
    # 获取模型配置
    model_cfg = cfgs['model_cfg']
    # 记录模型配置信息
    msg_mgr.log_info(model_cfg)
    # 从 models 模块中获取指定名称的模型类
    Model = getattr(models, model_cfg['model'])
    # 实例化模型
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        # 如果是训练阶段且配置了同步批归一化，则将模型中的批归一化层转换为同步批归一化层
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        # 如果配置了固定批归一化层，则固定模型中的批归一化层
        model.fix_BN()
    # 将模型包装为分布式数据并行模型
    # model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])

    model = model.cuda()

    # 记录模型的参数数量
    msg_mgr.log_info(params_count(model))
    # 记录模型初始化完成的信息
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        # 如果是训练阶段，调用模型的训练方法
        Model.run_train(model)
    else:
        # 如果是测试阶段，调用模型的测试方法
        Model.run_test(model)


if __name__ == '__main__':
    # # 初始化分布式训练环境，使用 NCCL 后端，通过环境变量进行初始化
    # torch.distributed.init_process_group('nccl', init_method='env://')
    # # 检查可用的 GPU 数量是否与分布式训练的世界大小（进程数量）相等，如果不相等则抛出异常
    # if torch.distributed.get_world_size() != torch.cuda.device_count():
    #     raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
    #         torch.cuda.device_count(), torch.distributed.get_world_size()))

    # 确保使用单GPU
    torch.cuda.set_device(0)

    # 加载配置文件
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        # 如果指定了恢复的迭代次数，则更新评估器和训练器的恢复提示
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    # 判断当前是否处于训练阶段
    training = (opt.phase == 'train')
    # 调用初始化函数进行初始化
    initialization(cfgs, training)
    # 调用运行模型的函数执行训练或测试操作
    run_model(cfgs, training)