# 导入必要的库和模块
import os                               # 用于文件和目录操作
import sys                              # 用于与系统交互（如添加路径）
import time                             # 用于时间测量
import logging                          # 用于记录日志
import json                             # 用于处理JSON数据
import random                           # 用于生成随机数
from collections import OrderedDict     # 用于保持字典项的顺序
import torch                            # 用于使用PyTorch框架
import numpy as np                      # 用于数值计算
import callback                         # 用于处理回调函数（自定义模块）

# 将'lib'目录插入系统路径，以便访问自定义库
sys.path.insert(0, '../lib')


class model(object):
    """
    代表深度学习模型的类，用于训练、验证、测试以及管理检查点。
    """

    def __init__(self, net, criterion, step_callback=None, step_callback_freq=50, epoch_callback=None,
                 save_checkpoint_freq=1, logger=None):
        """
        初始化模型，包含神经网络、损失函数、以及每一步和每个epoch的可选回调函数。
        """
        # 初始化关键参数
        self.net = net  # 网络模型
        self.criterion = criterion  # 损失函数
        self.step_callback_freq = step_callback_freq  # 步骤回调的频率
        self.save_checkpoint_freq = save_checkpoint_freq  # 保存检查点的频率
        self.logger = logger  # 日志记录器

        # 定义回调函数的参数
        self.callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None,
                                'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None, 'epoch_num': None,
                                'prefix': None}
        self.epoch_callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None,
                                      'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None, 'epoch_num': None,
                                      'prefix': 'Final'}

        # 如果没有传递step_callback，使用默认的速度监控和度量打印器
        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(), callback.MetricPrinter())
        # 如果没有传递epoch_callback，使用默认的度量打印器
        if not epoch_callback:
            epoch_callback = callback.CallbackList(callback.MetricPrinter())

        # 设置步骤回调和epoch回调
        self.step_callback = step_callback
        self.epoch_callback = epoch_callback

    def step_end_callback(self):
        """
        在每个步骤结束时执行的回调函数。
        """
        self.step_callback(**(self.callback_kwargs))  # 执行步骤回调

    def epoch_end_callback(self):
        """
        在每个epoch结束时执行的回调函数。
        """
        # 执行epoch回调
        self.epoch_callback(**(self.callback_kwargs))

        # 如果存在epoch耗时，记录日志
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Final_Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                self.callback_kwargs['epoch'], self.callback_kwargs['epoch_elapse'],
                self.callback_kwargs['epoch_elapse'] / 3600.))

        # 最终epoch回调
        self.epoch_callback(**(self.epoch_callback_kwargs))

        # 根据检查点频率决定是否保存模型
        if self.callback_kwargs['epoch'] == 0 or ((self.callback_kwargs['epoch'] + 1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch'] + 1,
                                 optimizer_state=self.callback_kwargs['optimizer_dict'])

    def load_state(self, state_dict, strict=False):
        """
        加载模型的状态字典。
        """
        if strict:
            # 严格加载模型的状态字典
            self.net.load_state_dict(state_dict=state_dict)
        else:
            # 自定义的部分加载函数
            net_state_keys = list(self.net.state_dict().keys())  # 当前模型的状态字典的键列表
            for name, param in state_dict.items():
                # 如果预训练模型中的参数在当前模型中，且形状匹配，则加载
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)  # 从键列表中移除已加载的参数
            # 记录未加载的层
            if net_state_keys:
                num_batches_list = []
                for i in range(len(net_state_keys)):
                    if 'num_batches_tracked' in net_state_keys[i]:
                        num_batches_list.append(net_state_keys[i])
                pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
                logging.info("当前网络中有未通过预训练初始化的层")
                logging.warning(">> 加载失败的层: {}".format(pruned_additional_states))
                return False
        return True

    def load_checkpoint(self, epoch, optimizer=None):
        """
        加载指定epoch的模型检查点。
        """
        load_path = '../result/'  # 检查点存放路径
        assert os.path.exists(load_path), "加载失败: {} (文件不存在)".format(load_path)

        checkpoint = torch.load(load_path)  # 加载检查点

        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)  # 加载模型状态

        if optimizer:
            # 如果存在优化器的状态且所有参数匹配，加载优化器的状态
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("模型和优化器状态从`{}'恢复".format(load_path))
            else:
                logging.warning(">> 从`{}'加载优化器状态失败".format(load_path))
        else:
            logging.info("仅恢复模型状态: `{}'".format(load_path))

        # 检查检查点中的epoch信息
        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning(">> Epoch信息不一致: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, optimizer_state=None):
        """
        保存当前模型和优化器的状态到检查点。
        """
        save_path = self.get_checkpoint_path(epoch)  # 获取保存路径
        save_folder = os.path.dirname(save_path)  # 获取保存文件夹路径

        # 如果文件夹不存在，创建它
        if not os.path.exists(save_folder):
            logging.debug("创建目录 {}".format(save_folder))
            os.makedirs(save_folder)

        # 保存模型状态和（可选的）优化器状态
        if not optimizer_state:
            torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, save_path)
            logging.info("检查点（仅模型）保存到: {}".format(save_path))
        else:
            torch.save({'epoch': epoch, 'state_dict': self.net.state_dict(), 'optimizer': optimizer_state}, save_path)
            logging.info("检查点（模型和优化器）保存到: {}".format(save_path))

    def fit(self, data_iter, dataset, optimizer, lr_scheduler, metrics=None, epoch_start=0, epoch_end=10000):
        """
        训练模型的主函数，用于执行从 epoch_start 到 epoch_end 的训练流程。
        参数:
        - data_iter: 数据迭代器
        - dataset: 数据集对象
        - optimizer: 优化器对象
        - lr_scheduler: 学习率调度器
        - metrics: 度量指标对象（可选）
        - epoch_start: 开始的 epoch（默认为 0）
        - epoch_end: 结束的 epoch（默认为 10000）

        主要步骤:
        1. 确保使用 GPU 进行训练。
        2. 循环执行每个 epoch 的训练和验证，并在每个 epoch 结束时调整学习率。
        """
        # 检查是否支持 GPU 训练
        assert torch.cuda.is_available(), "仅支持 GPU 版本"

        # 设置训练总 epoch 数
        self.callback_kwargs['epoch_num'] = epoch_end

        # 初始化训练所需的对象
        self.data_iter = data_iter
        self.dataset = dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end

        # 初始化模型 ID
        self.model_id = None

        # 遍历每一个 epoch
        for i_epoch in range(epoch_start, epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            self.epoch_callback_kwargs['namevals'] = []
            epoch_start_time = time.time()

            ###########
            # 1] 训练阶段
            ###########
            # 重置训练数据集并调用训练函数
            self.dataset.reset('train')
            self.train()

            ###########
            # 2] 评估阶段
            ###########
            # 在指定的 epoch 间隔执行测试评估
            if (self.data_iter is not None) \
                    and ((i_epoch + 1) % max(1, int(self.save_checkpoint_freq / 2))) == 0:
                self.dataset.reset('test')
                self.test()

            ###########
            # 3] Epoch 结束处理
            ###########
            # 更新学习率
            self.lr_scheduler.step()

            # 记录 epoch 的耗时并执行回调函数
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            self.epoch_end_callback()

        # 训练结束，记录日志
        self.logger.info("Optimization done!")

    def train(self):
        """
        模型的训练函数。
        主要步骤:
        1. 重置度量指标。
        2. 逐批处理数据，进行前向传播、反向传播和参数更新。
        3. 计算并记录每批次的度量结果和速度。
        4. 调用回调函数更新训练状态。
        """
        self.metrics.reset()  # 重置度量指标
        self.net.train()  # 设置模型为训练模式

        sum_sample_inst = 0  # 累计处理的样本数
        sum_sample_elapse = 0.  # 样本处理时间
        sum_update_elapse = 0  # 参数更新时间
        batch_start_time = time.time()  # 记录批次开始时间

        self.callback_kwargs['prefix'] = 'Train'  # 设置回调函数前缀

        i = 0  # 批次数初始化

        for i_batch, dats in enumerate(self.data_iter):
            i += 1
            self.callback_kwargs['batch'] = i_batch

            update_start_time = time.time()
            # 前向传播与损失计算
            outputs, losses = self.forward(dats)

            # 反向传播与参数更新
            self.optimizer.zero_grad()
            for loss in losses:
                loss.backward()
            self.optimizer.step()

            # 更新度量指标
            preds = [outputs[0][0].argmax(dim=-1).cpu().numpy()]
            mse_loss = torch.pow((preds[0] - dats[2].cpu()), 2).mean()
            self.metrics.update(preds, dats[2].cpu(), mse_loss, [loss.data.cpu() for loss in losses])

            sum_sample_elapse += time.time() - batch_start_time
            sum_update_elapse += time.time() - update_start_time
            batch_start_time = time.time()
            sum_sample_inst += dats[0].shape[0]

            # 每隔指定步数调用回调函数
            if (i_batch % self.step_callback_freq) == 0:
                self.callback_kwargs['namevals'] = self.metrics.get_name_value()
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                sum_update_elapse = 0
                sum_sample_elapse = 0
                sum_sample_inst = 0
                self.step_end_callback()

        # 最终度量结果
        self.callback_kwargs['namevals'] = self.metrics.get_name_value()
        self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
        self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
        self.step_end_callback()
        self.epoch_callback_kwargs['namevals'] += [[('Train_' + x[0][0], x[0][1])] for x in
                                                   self.metrics.get_name_value()]

    def val(self):
        """
        模型的验证函数。
        主要步骤:
        1. 重置度量指标。
        2. 遍历验证集进行前向传播，并计算度量指标。
        3. 记录每批次的时间与度量结果。
        """
        self.metrics.reset()
        self.net.eval()  # 设置模型为评估模式

        sum_sample_inst = 0
        sum_sample_elapse = 0.
        sum_update_elapse = 0
        batch_start_time = time.time()
        self.callback_kwargs['prefix'] = 'Val'

        for i_batch, dats in enumerate(self.data_iter):
            self.callback_kwargs['batch'] = i_batch
            update_start_time = time.time()

            outputs, losses = self.forward(dats)

            preds = [dats[2][outputs[0][0].argmax(dim=-1).cpu().numpy()]]
            mse_loss = torch.pow((preds[0] - dats[2].cpu()), 2).mean()
            self.metrics.update(preds, dats[2].cpu(), mse_loss)

            sum_sample_elapse += time.time() - batch_start_time
            sum_update_elapse += time.time() - update_start_time
            batch_start_time = time.time()
            sum_sample_inst += dats[0].shape[0]

            if (i_batch % self.step_callback_freq) == 0:
                self.callback_kwargs['namevals'] = self.metrics.get_name_value()
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                sum_update_elapse = 0
                sum_sample_elapse = 0
                sum_sample_inst = 0
                self.step_end_callback()

        self.callback_kwargs['namevals'] = self.metrics.get_name_value()
        self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
        self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
        self.step_end_callback()
        self.epoch_callback_kwargs['namevals'] += [[('Val_' + x[0][0], x[0][1])] for x in
                                                   self.metrics.get_name_value()]

    def test(self):
        """
        模型的测试函数，用于评估最终性能。
        主要步骤:
        1. 重置度量指标。
        2. 执行前向传播并计算度量指标和损失。
        3. 记录每批次的时间和度量结果。
        """
        self.metrics.reset()
        self.net.eval()

        sum_sample_inst = 0
        sum_sample_elapse = 0.
        sum_update_elapse = 0
        batch_start_time = time.time()
        self.callback_kwargs['prefix'] = 'Test'

        for i_batch, dats in enumerate(self.data_iter):
            self.callback_kwargs['batch'] = i_batch
            update_start_time = time.time()

            outputs, losses = self.forward(dats)

            preds = torch.nn.functional.softmax(outputs[0][2], dim=1)
            sorted_preds, indices = torch.sort(preds)
            cumsum_preds = torch.cumsum(sorted_preds, dim=1)
            preds_mask = torch.where(cumsum_preds <= 0.1, 0.0, 1.0)
            final_preds = (preds_mask * indices * sorted_preds).sum(dim=1, keepdim=True) / (
                    ((preds_mask * sorted_preds).sum(dim=1, keepdim=True)) * 125.0)
            final_preds = [final_preds.cpu()]

            mse_loss = torch.pow((final_preds[0] - dats[2].cpu()), 2).mean()
            self.metrics.update(final_preds, dats[2].cpu(), mse_loss, [loss.data.cpu() for loss in losses])

            sum_sample_elapse += time.time() - batch_start_time
            sum_update_elapse += time.time() - update_start_time
            batch_start_time = time.time()
            sum_sample_inst += dats[0].shape[0]

            if (i_batch % self.step_callback_freq) == 0:
                self.callback_kwargs['namevals'] = self.metrics.get_name_value()
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                sum_update_elapse = 0
                sum_sample_elapse = 0
                sum_sample_inst = 0
                self.step_end_callback()

        self.callback_kwargs['namevals'] = self.metrics.get_name_value()
        self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
        self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
        self.step_end_callback()
        self.epoch_callback_kwargs['namevals'] += [[('Test_' + x[0][0], x[0][1])] for x in
                                                   self.metrics.get_name_value()]

    def forward(self, dats):
        """
        前向传播函数。
        根据训练/评估状态执行前向传播和损失计算。
        参数:
        - dats: 包含输入数据、目标和其他辅助信息的批次
        返回:
        - output: 模型的输出
        - loss: 计算的损失
        """
        val = False
        if self.net.training:
            torch.set_grad_enabled(True)
            input_x = dats[0].float().cuda()
            target_var = dats[1].float().cuda()
            input_hff = dats[2].float().cuda()
            prompt = dats[3].float().cuda()
        else:
            torch.set_grad_enabled(False)
            val = True
            with torch.no_grad():
                input_x = dats[0].float().cuda()
                target_var = dats[1].float().cuda()
                input_hff = dats[2].float().cuda()
                prompt = dats[3].float().cuda()

        output = self.net.forward(input_x, input_hff, prompt, val=val)
        loss = self.criterion(output, target_var)

        return [output], [loss]

