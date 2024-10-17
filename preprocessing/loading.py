import numpy as np
import pandas as pd


# 测试集比例
RATIO = 0.3
# 随机种子
RANDOM_SEED = 42


# 定义一个函数用于将数据集划分为训练集和测试集
def train_test_split(X, y, hff, test_ratio=0.3, seed=None):
    # 检查样本和标签个数是否一致
    assert X.shape[0] == y.shape[0], '样本和标签个数不一致'

    # 检查测试比例是否合理
    assert 0 <= test_ratio < 1, '无效的测试比例'
    if seed:
        # 设置随机种子
        np.random.seed(seed)

    # 对索引进行洗牌
    shuffled_indexes = np.random.permutation(len(X))

    # 计算测试集大小
    test_size = int(len(X) * test_ratio)

    # 划分训练集和测试集的索引
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]

    # 返回划分后的训练集和测试集
    return X[train_index], X[test_index], hff[train_index], hff[test_index], y[train_index], y[test_index]


# 定义一个函数用于加载数据
def load_data(name):
    # 拼接文件路径
    dataPath = './dataset/' + name + '/dataSet_' + name
    labelPath = './dataset/' + name + '/labelSet_' + name
    featPath = './dataset/' + name + '/featSet_' + name

    # 加载原始数据与标签
    df = pd.read_csv(dataPath + '.csv').dropna()
    label_df = pd.read_csv(labelPath + '.csv').dropna()

    # 加载特征数据
    hff_df = pd.read_csv(featPath + '.csv').dropna()

    # 获取原始数据和标签
    raw_data = np.array(df.iloc[:, :])
    labels = np.array(label_df.iloc[:, -1])
    hff_data = np.array(hff_df.iloc[:, :])

    print(raw_data.shape)
    print(labels.shape)
    print(hff_data.shape)

    # 划分训练集和测试集
    X_train, X_test, X_train_H, X_test_H, y_train, y_test = train_test_split(raw_data, labels, hff_data, RATIO, RANDOM_SEED)

    # 将数据转换为适合神经网络的输入形式
    X_train_raw = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_raw = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_train_hff = X_train_H.reshape(X_train_H.shape[0], X_train_H.shape[1], 1)
    X_test_hff = X_test_H.reshape(X_test_H.shape[0], X_test_H.shape[1], 1)

    # 返回训练集和测试集
    return X_train_raw, X_test_raw, X_train_hff, X_test_hff, y_train, y_test

