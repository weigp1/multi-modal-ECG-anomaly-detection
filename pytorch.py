import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from collections import Counter
from torchmetrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.loading import load_data


# 定义Lightning模块
class BiLSTMModel(pl.LightningModule):
    def __init__(self, X_shape, HFF_shape):
        super(BiLSTMModel, self).__init__()

        # 原始流模型
        self.raw_lstm = nn.LSTM(input_size=X_shape[0], hidden_size=256, bidirectional=False)
        self.raw_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        # 特征流模型
        self.feat_lstm = nn.LSTM(input_size=HFF_shape[0], hidden_size=256, bidirectional=False)
        self.feat_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        # 合并两个流的输出
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.accuracy = Accuracy(task='multiclass', num_classes=5, average='macro')
        self.precision = Precision(task='multiclass', num_classes=5, average='macro')
        self.recall = Recall(task='multiclass', num_classes=5, average='macro')

    def forward(self, raw_input, feat_input):
        raw_out, _ = self.raw_lstm(raw_input.permute(0, 2, 1))
        raw_out = self.raw_fc(raw_out[:, -1, :])

        feat_out, _ = self.feat_lstm(feat_input.permute(0, 2, 1))
        feat_out = self.feat_fc(feat_out[:, -1, :])

        # 合并两个流的输出
        combined_out = self.avg_pooling(raw_out.unsqueeze(2) + feat_out.unsqueeze(2)).squeeze(2)

        return combined_out

    # 定义训练步骤
    def training_step(self, batch, batch_idx):
        # 从批次中获取训练数据
        X_train, X_train_H, y_train = batch

        # 前向传播获取模型输出
        output = self(X_train, X_train_H)

        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss()(output, y_train)

        # 返回损失
        return loss

    # 定义验证步骤
    def validation_step(self, batch, batch_idx):
        # 从批次中获取验证数据
        X_val, X_val_H, y_val = batch

        # 前向传播获取模型输出
        output = self(X_val, X_val_H)

        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss()(output, y_val)

        # 计算预测类别
        preds = torch.argmax(output, dim=1)

        # 计算准确率
        acc = self.accuracy(preds, y_val)

        # 记录验证损失和准确率
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        # 返回损失
        return loss

    # 定义测试步骤
    def test_step(self, batch, batch_idx):
        # 从批次中获取测试数据
        X_test, X_test_H, y_test = batch

        # 前向传播获取模型输出
        output = self(X_test, X_test_H)

        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss()(output, y_test)

        # 计算预测类别
        preds = torch.argmax(output, dim=1)

        # 计算准确率、精确度和召回率
        acc = self.accuracy(preds, y_test)
        pre = self.precision(preds, y_test)
        re = self.recall(preds, y_test)

        # 记录测试损失、准确率、精确度和召回率
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', acc, prog_bar=True)
        self.log('test_precision', pre, prog_bar=True)
        self.log('test_recall', re, prog_bar=True)

        # 返回损失
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    # 加载训练和测试数据集
    X_train, X_test, X_train_H, X_test_H, y_train, y_test = load_data('mitdb')

    # # 统计训练集中不同类别的样本数量
    # abnormal_types = Counter(y_train)
    # print("训练集中不同类别的样本数量：", abnormal_types.keys(), abnormal_types.values())
    #
    # # 统计测试集中不同类别的样本数量
    # abnormal_types = Counter(y_test)
    # print("测试集中不同类别的样本数量：", abnormal_types.keys(), abnormal_types.values())

    # 定义项目路径和模型路径
    project_path = "./models/"
    model_path = project_path + "ecg_model.pth"

    # 将数据转换为PyTorch张量
    X_train = torch.from_numpy(X_train).float()
    X_train_H = torch.from_numpy(X_train_H).float()
    y_train = torch.from_numpy(y_train).long()

    X_test = torch.from_numpy(X_test).float()
    X_test_H = torch.from_numpy(X_test_H).float()
    y_test = torch.from_numpy(y_test).long()

    # 创建训练集的DataLoader
    train_dataset = TensorDataset(X_train, X_train_H, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    print(X_train.shape[1]," ",X_train.shape[2])
    print(X_train_H.shape[1], " ", X_train_H.shape[2])

    # 初始化模型
    raw_input_shape = (X_train.shape[1], X_train.shape[2])
    hff_input_shape = (X_train_H.shape[1], X_train_H.shape[2])
    model = BiLSTMModel(raw_input_shape, hff_input_shape)

    # 创建PyTorch Lightning训练器
    trainer = pl.Trainer(max_epochs=30)

    # 如果已经存在保存的模型，则加载模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        # 训练模型
        trainer.fit(model, train_loader)
        # 保存训练好的模型
        torch.save(model.state_dict(), model_path)

    # 将模型设置为评估模式
    model.eval()

    # 创建测试集的DataLoader
    test_dataset = TensorDataset(X_test, X_test_H, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 使用测试集评估模型
    trainer.test(model, test_loader)

