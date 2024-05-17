from sentence_transformers import SentenceTransformer, util


def encode(text):
    # 加载预训练的模型
    model = SentenceTransformer('sentence-t5-base')

    # 计算文本的嵌入
    embedding = model.encode(text, convert_to_tensor=True)

    return embedding


def similarity(embedding1, embedding2):
    # 计算两个文本之间的余弦相似度
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)

    return cosine_similarity.item()

