# 导入标准库模块
import os
import argparse
import pprint

# 导入第三方库
import numpy as np
import random
import warnings
import torch

# 导入本地应用模块
from config import config
from lib import metric_clip as metric  # 自定义度量
from model_prompt import model  # 模型类
from dataloader.iterator_factory import get_dataiter  # 数据迭代器工厂
from network import bilstm_prompt
from lib import criterions  # 损失函数

# 禁用 cuDNN 以获得可重复性（性能较慢）
torch.backends.cudnn.enabled = False

if __name__ == "__main__":

    # 设置 CUDA 可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # 设置随机种子以保证结果的可复现性
    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('您已选择为训练设置随机种子。'
                      '这将开启 cuDNN 的确定性设置，'
                      '可能会显著降低您的训练速度！'
                      '从检查点重启时，您可能会看到意外的行为。')

    # 获取数据迭代器
    data_loader, data_iter = get_dataiter(config.data.root, config.data.set, config.train.batch_size,
                                          config.data.num_worker, 'clip')

    # 根据需要获取提示信息
    pmpt = getattr(data_iter, 'pmpt', None)

    sym_net = bilstm_prompt.BiLSTMModel(X_shape=250, HFF_shape=250,prompt_dict=pmpt)

    sym_net.float()  # 转换网络权重到 float 类型

    # 定义损失函数
    criterion = criterions.KLLoss_fast()

    # 创建模型实例
    net = model(net=sym_net, criterion=criterion, step_callback_freq=config.train.callback_freq,
                save_checkpoint_freq=config.save_frequency)

    net.net.cuda()  # 将模型移动到 GPU
    net.net = torch.nn.DataParallel(net.net).cuda()  # 使用多 GPU 并行处理

    # 根据配置选择优化器
    if config.train.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net.net.parameters(),
                                    lr=config.train.lr,
                                    momentum=0.9,
                                    weight_decay=0.0001,
                                    nesterov=True)
    elif config.train.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.net.parameters(),
                                     lr=config.train.lr,
                                     weight_decay=0.0001)
    elif config.train.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(net.net.parameters(),
                                      lr=config.train.lr,
                                      weight_decay=0.0001)
    elif config.train.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.net.parameters(),
                                        lr=config.train.lr,
                                        weight_decay=0.0001)
    else:
        raise NotImplementedError(config.train.optimizer.lower())  # 如果未实现的优化器被指定，则抛出异常

    # 设置学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x) for x in config.train.lr_epoch], gamma=config.train.lr_factor)

    # 定义评估指标
    metrics = metric.MetricList(metric.RMSE(), metric.CLIP_Loss())

    # 开始训练
    net.fit(data_iter=data_loader, dataset=data_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, metrics=metrics,
            epoch_start=0, epoch_end=config.train.end_epoch)
