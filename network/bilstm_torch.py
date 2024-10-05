import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from collections import Counter
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torch.utils.data import DataLoader, TensorDataset
from preprocessing.loading import load_data

# 定义一些全局变量
alpha = 0.3  # 用于控制两个流的输出权重的系数
batch_size = 128  # 批处理大小
type_nums = 5  # 类别数量


# 定义一个使用PyTorch Lightning框架的双向LSTM模型
class BiLSTMModel(pl.LightningModule):
    def __init__(self, X_shape, HFF_shape):
        super(BiLSTMModel, self).__init__()

        # 原始数据流的LSTM模型部分
        self.raw_lstm1 = nn.LSTM(input_size=X_shape[0], hidden_size=128, bidirectional=False, batch_first=True)
        self.raw_lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)

        # 原始数据流的全连接层
        self.raw_fc = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, out_features=type_nums),
            nn.Dropout(p=0.2)
        )

        # 特征数据流的全连接层
        self.feat_fc = nn.Sequential(
            nn.Linear(in_features=HFF_shape[0], out_features=16),
            nn.ReLU(),
            nn.Linear(16, out_features=type_nums),
            nn.Dropout(p=0.2)
        )

        # 平均池化层，用于整合LSTM的输出
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # 定义一些评估指标（多分类任务）
        self.accuracy = Accuracy(task='multiclass', num_classes=type_nums, average='macro')
        self.precision = Precision(task='multiclass', num_classes=type_nums, average='macro')
        self.recall = Recall(task='multiclass', num_classes=type_nums, average='macro')
        self.f1score = F1Score(task='multiclass', num_classes=type_nums, average='macro')
        self.auroc = AUROC(task='multiclass', num_classes=type_nums, average='macro')

    # 定义前向传播逻辑
    def forward(self, raw_input, feat_input):
        # 原始数据流的前向传播
        raw_out, _ = self.raw_lstm1(raw_input.permute(0, 2, 1))  # 调整输入的维度
        raw_out, _ = self.raw_lstm2(raw_out)
        raw_out = self.raw_fc(raw_out.squeeze(1))  # 经过全连接层并压缩维度
        raw_out = self.avg_pooling(raw_out.unsqueeze(2)).squeeze(2)

        # 特征数据流的前向传播
        feat_out = self.feat_fc(feat_input.squeeze(2))

        # 将两个流的输出进行加权合并
        combined_out = raw_out * alpha + feat_out * (1 - alpha)

        return combined_out

    # 定义训练步骤
    def training_step(self, batch, batch_idx):
        X_train, X_train_H, y_train = batch  # 从批次中获取数据
        output = self(X_train, X_train_H)  # 前向传播
        loss = nn.CrossEntropyLoss()(output, y_train)  # 计算损失
        self.log("train_loss", loss)  # 记录训练损失
        return loss

    # 定义验证步骤
    def validation_step(self, batch, batch_idx):
        X_val, X_val_H, y_val = batch  # 从批次中获取数据
        output = self(X_val, X_val_H)  # 前向传播
        loss = nn.CrossEntropyLoss()(output, y_val)  # 计算损失
        preds = torch.argmax(output, dim=1)  # 获取预测类别
        acc = self.accuracy(preds, y_val)  # 计算准确率
        self.log('val_loss', loss, prog_bar=True)  # 记录验证损失
        self.log('val_acc', acc, prog_bar=True)  # 记录验证准确率
        return loss

    # 定义测试步骤
    def test_step(self, batch, batch_idx):
        X_test, X_test_H, y_test = batch  # 获取测试数据
        output = self(X_test, X_test_H)  # 前向传播
        loss = nn.CrossEntropyLoss()(output, y_test)  # 计算损失

        # 计算多个评估指标
        acc = self.accuracy(output, y_test)
        pre = self.precision(output, y_test)
        re = self.recall(output, y_test)
        f1 = self.f1score(output, y_test)
        au = self.auroc(output, y_test)

        # 记录测试结果
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', acc, prog_bar=True)
        self.log('test_precision', pre, prog_bar=True)
        self.log('test_recall', re, prog_bar=True)
        self.log('test_f1score', f1, prog_bar=True)
        self.log('test_auroc', au, prog_bar=True)

        return loss

    # 配置优化器
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)


# 主函数入口
if __name__ == '__main__':
    # 加载训练和测试数据集
    X_train, X_test, X_train_H, X_test_H, y_train, y_test = load_data('mitdb')

    # 统计训练集和测试集中不同类别的样本数量
    abnormal_types = Counter(y_train)
    print("训练集中不同类别的样本数量：", abnormal_types.keys(), abnormal_types.values())
    abnormal_types = Counter(y_test)
    print("测试集中不同类别的样本数量：", abnormal_types.keys(), abnormal_types.values())

    # 定义模型存储路径
    project_path = "./models/"
    model_path = project_path + "model.pth"

    # 将数据转换为PyTorch张量
    X_train = torch.from_numpy(X_train).float()
    X_train_H = torch.from_numpy(X_train_H).float()
    y_train = torch.from_numpy(y_train).long()

    X_test = torch.from_numpy(X_test).float()
    X_test_H = torch.from_numpy(X_test_H).float()
    y_test = torch.from_numpy(y_test).long()

    # 创建训练集的DataLoader
    dataset = TensorDataset(X_train, X_train_H, y_train)
    train_size = int(0.7 * len(dataset))  # 划分训练集
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,
                              num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True,
                              num_workers=4)

    # 初始化模型
    raw_input_shape = (X_train.shape[1], X_train.shape[2])
    hff_input_shape = (X_train_H.shape[1], X_train_H.shape[2])
    model = BiLSTMModel(raw_input_shape, hff_input_shape)

    # 使用PyTorch Lightning的Trainer进行训练
    trainer = pl.Trainer(max_epochs=50, check_val_every_n_epoch=1)

    # 如果已有保存的模型则加载，否则训练新模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        torch.save(model.state_dict(), model_path)

    # 将模型设置为评估模式并进行测试
    model.eval()
    test_dataset = TensorDataset(X_test, X_test_H, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4)
    trainer.test(model, test_loader)
