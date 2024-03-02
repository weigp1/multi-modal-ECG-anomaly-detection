import os
import wfdb
import pywt
import seaborn
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 项目根目录
project_path = "./"
# 定义日志目录
# 必须是Web应用程序启动时指定的目录的子目录
# 推荐使用日期时间作为子目录名
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.h5"

# 测试集比例
RATIO = 0.3
# 随机种子
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30


# 导入数据集
def get_data_set(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 加载心电数据并去噪
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('../ECG/ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()

    # 获取R波位置和对应的标签
    annotation = wfdb.rdann('../ECG/ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去除首尾不稳定信号
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 选取特殊标签的数据(N/A/V/L/R), 其他舍去
    # X_data: R 波附近的260个数据点
    # Y_data: 将 N/A/V/L/R 映射为 0/1/2/3/4
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = data[Rlocation[i] - 80:Rlocation[i] + 180]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 导入数据，进行预处理
def load_data(ratio, random_seed):
    numberSet = ['100']
    """
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    """
    dataSet = []
    lableSet = []
    for n in numberSet:
        get_data_set(n, dataSet, lableSet)

    # 重排列 & 分割数据集
    dataSet = np.array(dataSet).reshape(-1, 260)
    lableSet = np.array(lableSet).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, lableSet, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test


def handcrafted_feat(data):  # 两种特征提取函数
    # 初始化特征数组
    fea = np.zeros((data.shape[0] - 1, 2 * data.shape[1]))
    # 创建时间步数组
    x = np.array(range(data.shape[0]))
    # 初始化线性回归模型
    regr = linear_model.LinearRegression()
    # 对于每条数据
    for i in range(data.shape[0] - 1):
        # 对于每个数据点
        for j in range(data.shape[1]):
            # 计算振幅平均特征
            fea[i, 2 * j] = np.mean(data[:(i + 2), j])
            # 使用线性回归拟合并计算线性模型的趋势系数特征
            regr.fit(x[:(i + 2)].reshape(-1, 1), np.ravel(data[:(i + 2), j]))
            fea[i, 2 * j + 1] = regr.coef_

    return fea


def main():
    # 训练集与测试集
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)

    # X_train.shape (1581, 260) 1581条数据，每条数据260个数据点
    print(X_train.shape)

    X_train_HFF = handcrafted_feat(X_train)
    X_test_HFF = handcrafted_feat(X_test)

    # X_train_HFF.shape (1580, 520)
    print(X_train_HFF.shape)


if __name__ == '__main__':
    main()
