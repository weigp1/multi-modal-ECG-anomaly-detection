from sentence_transformers import SentenceTransformer, util

# 加载预训练的模型（例如bert-base-nli-mean-tokens）
model = SentenceTransformer('bert-base-nli-mean-tokens')

# 输入文本
text1 = "这是一段示例文本。"
text2 = "这是另一段示例文本。"

# 计算文本的嵌入
embeddings1 = model.encode(text1, convert_to_tensor=True)
embeddings2 = model.encode(text2, convert_to_tensor=True)

# 计算两个文本之间的余弦相似度
cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

print(f"文本1和文本2之间的余弦相似度：{cosine_similarity.item()}")
