import numpy as np
import pywt
import wfdb
import matplotlib.pyplot as plt
from spectrum import arburg
import statsmodels.api as sm
from scipy.signal import lfilter


def AIC(data, order):
    # 拟合 AR 模型
    model = sm.tsa.AutoReg(data, lags=order)
    ar_model = model.fit()

    # 返回 AIC 值
    return ar_model.aic


# def AIC_burg(data, order=32):
#     # 计算自回归（AR）系数
#     coeffs, sigma2 = sm.regression.linear_model.burg(data, order)
#
#     # 计算模型拟合的残差
#     residuals = sm.tsa.AR(data).fit(order).resid
#
#     # 计算 AIC
#     n = len(data)
#     aic = n * (1 + (2 * order) / n) * (sigma2 / n)
#
#     return aic


def Plot_AIC(orders, aic_values):
    # 绘制AIC变化图
    plt.plot(orders, aic_values, marker='o')
    plt.title('AIC vs. AR Model Order')
    plt.xlabel('AR Model Order')
    plt.ylabel('AIC')
    plt.grid(True)
    plt.show()


# 定义函数以获取每个数据集的 AIC 最低的阶数和频次
def get_lowest_aic_order(data):
    lowest_aic_order = []
    for i in range(len(data)):
        aic_values = [AIC(data[i], order) for order in range(1, 61)]
        lowest_aic_order.append(np.argmin(aic_values) + 1)  # 记录最小AIC对应的阶数
    return lowest_aic_order


# 实现小波去噪函数
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


# 读取ECG数据和标签
def get_data_set(number, X_train):
    # 加载心电数据并去噪
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('D:/pycharm/PyCode/ECG/ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    data = denoise(data)

    # 获取R波位置和对应的标签
    annotation = wfdb.rdann('D:/pycharm/PyCode/ECG/ecg_data/' + number, 'atr')
    Rlocation = annotation.sample

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
            # 获得原始数据
            x_train = data[Rlocation[i] - 80:Rlocation[i] + 180]
            X_train.append(x_train)
            i += 1
        except ValueError:
            i += 1
    return X_train


# 导入数据，进行预处理
def load_data():
    numberSet = ['100']
    dataSet = []
    for n in numberSet:
        get_data_set(n, dataSet)

    # 重排列 & 分割数据集
    dataSet = np.array(dataSet).reshape(-1, 260)

    return dataSet


# 计算AIC
data = load_data()
orders = range(1, 61)
aic_values = [AIC(data[1], order) for order in orders]
print(aic_values)
Plot_AIC(orders, aic_values)
