import pandas as pd
import wfdb
import pywt
import numpy as np
from sklearn import linear_model

name = "mitdb"


def denoise(data):
    # 使用小波变换对信号进行分解
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 使用软阈值对分解后的系数进行去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 通过逆小波变换得到去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 导入数据集
def load_record(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 加载心电数据并去噪
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('E:/' + name + '/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()

    # 获取R波位置和对应的标签
    annotation = wfdb.rdann('E:/' + name + '/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去除首尾不稳定信号
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

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


if __name__ == '__main__':
    path = '../benchmarks/'
    numberSet = ['100']
    dataSet = []
    lableSet = []
    for n in numberSet:
        load_record(n, dataSet, lableSet)
    dataSet = np.array(dataSet).reshape(-1, 260)
    lableSet = np.array(lableSet).reshape(-1, 1)
    recordSet = np.hstack((dataSet, lableSet))
    df = pd.DataFrame(data=recordSet)
    df.to_csv(path + name + ".csv", index=False)
    # 提取手工特征
    print("handcrafting...")
    dataSet_H = handcrafted_feat(dataSet)
    np.save(path + name + ".npy", dataSet_H)
