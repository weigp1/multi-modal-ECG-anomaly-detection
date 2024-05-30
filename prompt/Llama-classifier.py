from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# 加载预训练的LLAMA模型和分词器
tokenizer = AutoTokenizer.from_pretrained("D:/study/ECG_Anomaly_Detection/llama-main/Llama-2-7b-hf", use_fast=True,
                                          padding=True, pad_token="[PAD]")
model = AutoModelForCausalLM.from_pretrained("D:/study/ECG_Anomaly_Detection/llama-main/Llama-2-7b-hf").to('cuda')


# 定义一个简单的线性分类器
class LlamaClassifier(nn.Module):
    def __init__(self, num_labels):
        super(LlamaClassifier, self).__init__()
        self.llama = model
        self.classifier = nn.Linear(model.config.n_embd, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


# 定义数据集
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(),
                "label": label}


# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# 创建数据集和数据加载器
texts = ["This movie is great!", "I hated this movie."]
labels = [1, 0]
dataset = TextClassificationDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 3

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')
        labels = batch["label"].to('cuda')

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # 清理未使用的缓存


# 评估模型
def evaluate(model, dataloader):

    model.eval()
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            labels = batch["label"].to('cuda')

            with torch.cuda.amp.autocast():  # 使用混合精度
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)
                total_acc += (predicted == labels).sum().item()
                total_samples += labels.size(0)

    accuracy = total_acc / total_samples
    return accuracy


# 测试数据集和数据加载器
test_texts = ["This movie is amazing!", "I didn't like this film."]
test_labels = [1, 0]
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 评估
accuracy = evaluate(model, test_dataloader)
print(f"Test Accuracy: {accuracy:.4f}")
