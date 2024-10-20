import torch
import numpy as np
import torch.nn as nn

from config import config


# 定义使用 PyTorch Lightning 框架的双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, raw_input_dim, feat_input_dim, prompt_dict):
        super(BiLSTMModel, self).__init__()

        # 将提示字典转换为浮点数张量并移到 GPU 上
        self.prompt_dict = torch.tensor(prompt_dict.values).float().cuda()

        # 导入模型参数
        raw_hidden_dim = config.net.raw_hidden_dim
        raw_linear_dim = config.net.raw_linear_dim
        raw_hidden_dim_2 = config.net.raw_hidden_dim_2
        feat_hidden_dim = config.net.feat_hidden_dim
        feat_linear_dim = config.net.feat_linear_dim
        output_dim = config.net.output_dim
        prompt_dim = config.net.prompt_dim
        dropout_rate = config.net.dropout

        # 原始数据流 LSTM 模型
        self.raw_lstm = nn.LSTM(
            input_size=raw_input_dim,
            hidden_size=raw_hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.raw_lstm_2 = nn.LSTM(
            input_size=2*raw_hidden_dim,
            hidden_size=raw_hidden_dim_2,
            bidirectional=True,
            batch_first=True,
            dropout = dropout_rate,
        )

        # 原始数据流的全连接层
        self.raw_fc = nn.Sequential(
            nn.Linear(in_features=2*raw_hidden_dim_2, out_features=raw_linear_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=raw_linear_dim, out_features=output_dim),
            nn.ReLU(inplace=True),
        )

        # 特征数据流 LSTM 模型
        # self.hff_lstm = nn.LSTM(
        #     input_size=feat_input_dim,
        #     hidden_size=feat_hidden_dim,
        #     bidirectional=True,
        #     batch_first=True,
        #     dropout=dropout_rate,
        # )

        # 特征数据流的全连接层
        self.feat_fc = nn.Sequential(
            nn.Linear(in_features=feat_input_dim, out_features=feat_linear_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feat_linear_dim, out_features=output_dim),
            nn.ReLU(inplace=True),
        )
        
        # 平均池化层
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # 定义提示投影层
        self.pmp_proj1 = nn.Linear(in_features=prompt_dim, out_features=output_dim)     # 提示特征投影
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))              # 可学习的标量参数


    # 定义前向传播逻辑
    def forward(self, raw_input, feat_input, prompt, val=False):

        # print(raw_input.size())       # torch.Size([4, 250]), batch_size=4, len=250

        # 原始数据流LSTM层
        raw_input = torch.tensor(raw_input, dtype=torch.float32).unsqueeze(1)   # torch.Size([4, 1, 250])
        raw_out = self.raw_lstm(raw_input)[0]
        raw_out = self.raw_lstm_2(raw_out)[0]

        # 线性变化
        hidden_dim = raw_out.shape[-1]
        raw_out = raw_out.reshape(-1, hidden_dim)
        raw_out = self.raw_fc(raw_out)

        # L2 归一化
        raw_out = raw_out / raw_out.norm(dim=1, keepdim=True)

        # 特征数据流LSTM层
        # feat_input = torch.tensor(feat_input, dtype=torch.float32).unsqueeze(1)
        # feat_out = self.hff_lstm(feat_input)[0]

        # 线性变化
        # hidden_dim = feat_out.shape[-1]
        # feat_out = feat_out.reshape(-1, hidden_dim)
        feat_out = self.feat_fc(feat_input)

        # L2 归一化
        feat_out = feat_out / feat_out.norm(dim=1, keepdim=True)

        # 合并双流输出
        alpha = config.net.alpha
        combined_out = alpha * raw_out + (1-alpha) * feat_out

        # 提示特征投影
        pmp_feat = self.pmp_proj1(prompt)
        pmp_feat = pmp_feat / pmp_feat.norm(dim=1, keepdim=True)

        # 计算 logits
        logit_scale = self.logit_scale.exp()                        # 应用 exp 函数
        logits_per_x = logit_scale * combined_out @ pmp_feat.t()    # 计算 logits
        logits_per_pmp = logits_per_x.t()

        # 如果是验证阶段，计算预测 logits
        if val:
            preds_feat = self.pmp_proj1(self.prompt_dict)                   # 投影提示字典
            preds_feat = preds_feat / preds_feat.norm(dim=1, keepdim=True)  # 归一化
            logits_per_pred = logit_scale * combined_out @ preds_feat.t()   # 计算文本到预测的 logits
            return logits_per_x, logits_per_pmp, logits_per_pred

        # 返回 outputs 和 losses
        return logits_per_x, logits_per_pmp
