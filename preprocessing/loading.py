import numpy as np
import pandas as pd

# 测试集比例
RATIO = 0.3
# 随机种子
RANDOM_SEED = 42


def train_test_split(X, HFF, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], '样本和标签个数不一致'
    assert 0 <= test_ratio < 1, '无效的测试比例'
    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X[train_index], X[test_index], HFF[train_index], HFF[test_index], y[train_index], y[test_index]


def load_data(name):
    path = './benchmarks/' + name
    HFF = np.load(path + '.npy')

    df = pd.read_csv(path + '.csv').dropna().drop_duplicates()
    raw_data = np.array(df.iloc[:, :-1])
    labels = np.array(df.iloc[:, -1])

    X_train, X_test, X_train_H, X_test_H, y_train, y_test = train_test_split(raw_data, HFF, labels, RATIO, RANDOM_SEED)

    # 转换为适合神经网络的输入形式
    X_train_raw = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_raw = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_train_hff = X_train_H.reshape(X_train_H.shape[0], X_train_H.shape[1], 1)
    X_test_hff = X_test_H.reshape(X_test_H.shape[0], X_test_H.shape[1], 1)

    return X_train_raw, X_test_raw, X_train_hff, X_test_hff, y_train, y_test
