from datetime import datetime

import pandas as pd
import wfdb
import pywt
import numpy as np
from sklearn import linear_model
from concurrent.futures import ThreadPoolExecutor

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


def handcrafted_feat(data):
    rows, cols = data.shape
    fea = np.zeros((rows - 1, 2 * cols))

    x = np.arange(rows)
    regr = linear_model.LinearRegression()

    for i in range(rows - 1):
        subset_data = data[:(i + 2)]

        # 计算振幅平均特征
        fea[i, :cols] = np.mean(subset_data, axis=0)

        # 使用线性回归拟合并计算线性模型的趋势系数特征
        for j in range(cols):
            regr.fit(x[:(i + 2)].reshape(-1, 1), subset_data[:, j])
            fea[i, cols + j] = regr.coef_

    return fea


# #   并行化
# def calculate_feature(i, j, x, subset_data):
#     regr = linear_model.LinearRegression()
#     regr.fit(x.reshape(-1, 1), subset_data[:, j])
#     return regr.coef_
#
#
# def process_row(i, data, x, cols):
#     fea_row = np.zeros(2 * cols)
#
#     subset_data = data[:(i + 2)]
#
#     # 计算振幅平均特征
#     fea_row[:cols] = np.mean(subset_data, axis=0)
#
#     # 并行计算线性模型的趋势系数特征
#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(calculate_feature, i, j, x[:(i + 2)], subset_data)
#             for j in range(cols)
#         ]
#
#         # 获取并行计算的结果
#         for j, future in enumerate(futures):
#             fea_row[cols + j] = future.result()
#
#     return fea_row
#
#
# def handcrafted_feat(data):
#     rows, cols = data.shape
#     fea = np.zeros((rows - 1, 2 * cols))
#
#     x = np.arange(rows)
#
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_row, i, data, x, cols) for i in range(rows - 1)]
#
#         # 获取并行计算的结果
#         for i, future in enumerate(futures):
#             fea[i] = future.result()
#
#     return fea


if __name__ == '__main__':

    path = '../benchmarks/'
    numberSet = ['100']
    # numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
    #              '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
    #              '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
    #              '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        load_record(n, dataSet, lableSet)
    dataSet = np.array(dataSet).reshape(-1, 260)
    lableSet = np.array(lableSet).reshape(-1, 1)
    recordSet = np.hstack((dataSet, lableSet))
    df = pd.DataFrame(data=recordSet)
    # 提取手工特征
    time1 = datetime.now()
    print("handcrafting...")
    dataSet_H = handcrafted_feat(dataSet)
    time2 = datetime.now()
    print((time2 - time1).seconds)
    df.to_csv(path + name + ".csv", index=False)
    np.save(path + name + ".npy", dataSet_H)
