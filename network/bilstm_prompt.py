import torch
import numpy as np
import torch.nn as nn


# 定义使用PyTorch Lightning框架的双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, X_shape, HFF_shape, prompt_dict):
        super(BiLSTMModel, self).__init__()

        self.prompt_dict = torch.tensor(prompt_dict.values).float().cuda()  # 将提示字典转换为浮点数张量并移到 GPU 上

        # 原始LSTM模型
        self.raw_lstm = nn.LSTM(
            input_size=X_shape,
            hidden_size=128, 
            bidirectional=True, 
            batch_first=True,
            dropout=0,
        )

        self.hff_lstm = nn.LSTM(
            input_size=HFF_shape,
            hidden_size=128, 
            bidirectional=True, 
            batch_first=True,
            dropout=0,
        )

        # 原始数据流的全连接层
        self.raw_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, out_features=8)
        )

        # 特征数据流的全连接层
        self.feat_fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(64, out_features=8)
        )
        
        # 平均池化层
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # 定义提示投影层
        self.pmp_proj1 = nn.Linear(4096, 8)  # 线性层用于提示特征投影
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 可学习的标量参数


    # 定义前向传播逻辑
    def forward(self, raw_input, feat_input, prompt, val=False):

        # print(raw_input.size())       # torch.Size([4, 250]), batch_size=4, len=250

        # 原始数据流的前向传播
        raw_input = torch.tensor(raw_input, dtype=torch.float32).unsqueeze(1)
        raw_out, _ = self.raw_lstm(raw_input)
        raw_out = self.raw_fc(raw_out.squeeze(1))  # 经过全连接层并压缩维度
        # raw_out = self.avg_pooling(raw_out.unsqueeze(2))

        # 对输出进行 L2 归一化
        raw_out = raw_out / raw_out.norm(dim=1, keepdim=True)

        # 特征数据流的前向传播
        feat_input = torch.tensor(feat_input, dtype=torch.float32).unsqueeze(1)
        feat_out, _ = self.hff_lstm(feat_input)
        feat_out = self.feat_fc(feat_out.squeeze(1))
        # feat_out = self.avg_pooling(feat_out.unsqueeze(2))

        # 对输出进行 L2 归一化
        feat_out = feat_out / feat_out.norm(dim=1, keepdim=True)

        alpha = 0.3
        combined_out = alpha * raw_out + (1 - alpha) * feat_out

        # 提示特征投影
        pmp_feat = self.pmp_proj1(prompt)
        pmp_feat = pmp_feat / pmp_feat.norm(dim=1, keepdim=True)

        # 计算 logits
        logit_scale = self.logit_scale.exp()  # 应用 exp 函数
        logits_per_x = logit_scale * combined_out @ pmp_feat.t()  # 计算 logits
        logits_per_pmp = logits_per_x.t()

        # 如果是验证阶段，计算预测 logits
        if val:
            preds_feat = self.pmp_proj1(self.prompt_dict)  # 投影提示字典
            preds_feat = preds_feat / preds_feat.norm(dim=1, keepdim=True)  # 归一化
            logits_per_pred = logit_scale * combined_out @ preds_feat.t()  # 计算文本到预测的 logits
            return logits_per_x, logits_per_pmp, logits_per_pred

        # 返回 logits
        return logits_per_x, logits_per_pmp

    


