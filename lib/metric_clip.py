import logging
import numpy as np


####################
# COMMON METRICS
####################

class EvalMetric(object):
    """
    评估指标基类，定义了所有评估指标的公共接口。
    """

    def __init__(self, name, **kwargs):
        """
        初始化评估指标。

        Args:
            name (str): 指标名称。
            **kwargs: 其他关键字参数。
        """
        self.name = str(name)  # 设置指标名称
        self.reset()  # 初始化指标

    def update(self, preds, labels, losses):
        """
        更新评估指标。

        Args:
            preds: 模型预测结果。
            labels: 真实标签。
            losses: 损失值。

        Raises:
            NotImplementedError:  子类必须实现此方法。
        """
        raise NotImplementedError()  # 此方法必须由子类实现

    def reset(self):
        """重置评估指标."""
        self.num_inst = 0  # 实例数量
        self.sum_metric = 0.0  # 指标累加值

    def get(self):
        """
        获取评估指标的值。

        Returns:
            tuple: (指标名称, 指标值)。如果实例数量为0，则返回 NaN。
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))  # 如果没有实例，返回 NaN
        else:
            return (self.name, self.sum_metric / self.num_inst)  # 计算并返回平均指标值

    def get_name_value(self):
        """
        获取指标名称和值的列表。

        Returns:
            list:  [(指标名称, 指标值)] 的列表。
        """
        name, value = self.get()  # 获取指标名称和值
        if not isinstance(name, list):  # 确保名称是列表
            name = [name]
        if not isinstance(value, list):  # 确保值是列表
            value = [value]
        return list(zip(name, value))  # 返回名称和值的列表

    def check_label_shapes(self, preds, labels):
        """
        检查预测结果和标签的形状是否一致。

        Args:
            preds: 模型预测结果。
            labels: 真实标签。

        Raises:
            NotImplementedError: 如果形状不一致。
        """
        if (type(labels) is list) and (type(preds) is list):
            label_shape, pred_shape = len(labels), len(preds)
        else:
            label_shape, pred_shape = labels.shape[0], preds.shape[0]

        if label_shape != pred_shape:
            raise NotImplementedError("Prediction and label shapes are inconsistent.")


class MetricList(EvalMetric):
    """
    处理多个评估指标。
    """

    def __init__(self, *args, name="metric_list"):
        """
        初始化 MetricList。

        Args:
            *args:  多个 EvalMetric 对象。
            name (str): 指标列表的名称，默认为 "metric_list"。

        Raises:
            AssertionError: 如果输入的不是 EvalMetric 对象。
        """
        assert all([issubclass(type(x), EvalMetric) for x in args]), \
            "MetricList input is illegal: {}".format(args)  # 断言输入必须是 EvalMetric 对象
        self.metrics = [metric for metric in args]  # 存储多个指标
        super(MetricList, self).__init__(name=name)  # 调用父类的初始化方法

    def update(self, preds, labels, losses=None, clip_loss=None):
        """
        更新所有指标。

        Args:
            preds: 模型预测结果。
            labels: 真实标签。
            losses: 损失值。
            clip_loss: 裁剪损失值。
        """
        preds = [preds] if type(preds) is not list else preds  # 确保 preds 是列表
        labels = [labels] if type(labels) is not list else labels  # 确保 labels 是列表
        losses = [losses] if type(losses) is not list else losses  # 确保 losses 是列表
        clip_loss = [clip_loss] if type(clip_loss) is not list else clip_loss  # 确保 clip_loss 是列表

        for metric in self.metrics:  # 遍历所有指标
            metric.update(preds, labels, losses, clip_loss)  # 更新每个指标

    def reset(self):
        """重置所有指标."""
        if hasattr(self, 'metrics'):  # 检查是否定义了指标
            for metric in self.metrics:  # 遍历所有指标
                metric.reset()  # 重置每个指标
        else:
            logging.warning("No metric defined.")  # 如果没有定义指标，则发出警告

    def get(self):
        """
        获取所有指标的值。

        Returns:
            list:  [(指标名称, 指标值)] 的列表。
        """
        ouputs = []
        for metric in self.metrics:  # 遍历所有指标
            ouputs.append(metric.get())  # 获取每个指标的值
        return ouputs

    def get_name_value(self):
        """
        获取所有指标的名称和值的列表。

        Returns:
            list:  [[(指标名称, 指标值)]] 的列表。
        """
        ouputs = []
        for metric in self.metrics:  # 遍历所有指标
            ouputs.append(metric.get_name_value())  # 获取每个指标的名称和值
        return ouputs


####################
# RUL EVAL METRICS
####################

class RMSE(EvalMetric):
    """
    均方根误差 (RMSE) 指标。
    """

    def __init__(self, name='RMSE'):
        """
        初始化 RMSE 指标。

        Args:
            max_rul: 最大剩余使用寿命。
            name (str): 指标名称，默认为 'RMSE'。
        """
        super(RMSE, self).__init__(name)  # 调用父类的初始化方法

    def update(self, preds, labels, losses, clip_loss=None):
        """
        更新 RMSE 指标。

        Args:
            preds: 模型预测结果。
            labels: 真实标签。
            losses: 损失值 (必须提供)。
            clip_loss: 裁剪损失值 (未使用)。
        """
        assert losses is not None, "Loss undefined."  # 断言损失值必须提供
        for loss in losses:  # 遍历所有损失值
            self.sum_metric += float(loss.numpy().sum()) * labels[0].shape[0]  # 累加损失的平方和
            self.num_inst += labels[0].shape[0]  # 更新实例数量

    def get(self):
        """
        获取 RMSE 值。

        Returns:
            tuple: (指标名称, RMSE 值)。
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))  # 如果没有实例，返回 NaN
        else:
            return (self.name, np.sqrt(self.sum_metric / self.num_inst))  # 计算并返回 RMSE 值


class CLIP_Loss(EvalMetric):
    """
    CLIP损失指标。
    """

    def __init__(self, name='CLIP_Loss'):
        """
        初始化 CLIP_Loss 指标。

        Args:
            max_rul: 最大剩余使用寿命 (未使用)。
            name (str): 指标名称，默认为 'CLIP_Loss'。
        """
        super(CLIP_Loss, self).__init__(name)  # 调用父类的初始化方法

    def update(self, preds, labels, losses, clip_loss):
        """
        更新 CLIP_Loss 指标。

        Args:
            preds: 模型预测结果 (未使用)。
            labels: 真实标签 (未使用)。
            losses: 损失值 (未使用)。
            clip_loss: 裁剪损失值 (必须提供)。
        """
        assert losses is not None, "Loss undefined."  # 断言损失值必须提供 (虽然未使用)
        for loss in clip_loss:  # 遍历所有裁剪损失值
            self.sum_metric += float(loss.numpy().sum()) * labels[0].shape[0]  # 累加裁剪损失
            self.num_inst += labels[0].shape[0]  # 更新实例数量

    def get(self):
        """
        获取 CLIP_Loss 值。

        Returns:
            tuple: (指标名称, CLIP_Loss 值)。
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))  # 如果没有实例，返回 NaN
        else:
            return (self.name, np.sqrt(self.sum_metric / self.num_inst))  # 计算并返回 CLIP_Loss 值