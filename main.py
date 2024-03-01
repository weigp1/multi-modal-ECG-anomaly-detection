import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


class LitSimpleCNN(pl.LightningModule):
    def __init__(self):
        super(LitSimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y, preds)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss


# 设置数据
dataset = MNIST(os.getcwd(), download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化PyTorch Lightning模型
model = LitSimpleCNN()

# 训练模型
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader)
