import wfdb
import pywt
import numpy as np
from scipy.stats import kurtosis, skew

from hfd import HFD
from matplotlib import pyplot as plt
from spectrum import arburg

# 测试集比例
RATIO = 0.3
# 随机种子
RANDOM_SEED = 42
# 数据标签
ecgClassSet = ['N', 'A', 'V', 'L', 'R']


# 香农小波熵
def SE(data):
    # 使用DB小波函数
    coeffs = pywt.wavedec(data, 'db1', level=4)

    # 计算能量概率分布
    s_coeffs = [np.square(np.abs(coeff)) for coeff in coeffs]
    total_energy = np.sum(np.concatenate(s_coeffs))
    probabilities = [s_coeff / total_energy for s_coeff in s_coeffs]

    # 计算小波香农熵
    entropy = 0
    for probability in probabilities:
        entropy -= np.sum(probability * np.log2(probability + 1e-10))

    return entropy


# 小波方差
def WV(data):
    # 小波分解
    coeffs = pywt.swt(data, 'db2', level=2, norm=True)

    # 计算小波方差
    wavelet_var = []
    for coeff in coeffs:
        variances = [(w ** 2).mean() for w in coeff]
        wavelet_var.extend(variances)

    return wavelet_var


# AR系数
def ARC(data, order=32):
    # 计算自回归（AR）系数
    coefficients, _, _ = arburg(data, order)
    coefficients = coefficients.real.astype(float)
    return coefficients


# 静态特征
def SF(data):
    # 五个静态特征分别为: 平均值、标准差、中位数、峰度、偏度
    Mean = np.mean(data)
    Std = np.std(data)
    Med = np.median(data)
    Kurt = kurtosis(data)
    Skew = skew(data)
    return np.array([Mean, Std, Med, Kurt, Skew])


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
def get_data_set(number, X_data, Y_data, featSet):
    # 加载心电数据并去噪
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('D:/myproject/ECG_Benchmark/data/test/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    data = denoise(data)

    # 获取R波位置和对应的标签
    annotation = wfdb.rdann('D:/myproject/ECG_Benchmark/data/test/' + number, 'atr')
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
            # 获得原始数据与标签
            lable = ecgClassSet.index(Rclass[i])
            x_train = data[Rlocation[i] - 80:Rlocation[i] + 180]

            X_data.append(x_train)
            Y_data.append(lable)

            feat_sf = np.array(SF(x_train))
            feat_ar = np.array(ARC(x_train))
            feat_se = np.array([SE(x_train)])
            feat_wv = np.array(WV(x_train))
            feat_hfd = np.array([HFD(x_train)])

            featSet.append(np.concatenate((feat_sf, feat_ar, feat_se, feat_wv, feat_hfd), axis=None))
            i += 1
        except ValueError:
            i += 1
    return


# 导入数据，进行预处理
def load_data():
    numberSet = ['100']
    dataSet = []
    lableSet = []
    featSet = []
    for n in numberSet:
        get_data_set(n, dataSet, lableSet, featSet)

    # 重排列 & 分割数据集
    dataSet = np.array(dataSet).reshape(-1, 260)
    lableSet = np.array(lableSet).reshape(-1)

    # SVD奇异值分解
    featSet = np.array(featSet)
    U, S, V = np.linalg.svd(featSet)
    k = 2  # 选择前2个奇异值作为主要特征
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    featSet_approx = np.dot(U_k, np.dot(S_k, V_k))

    print(featSet_approx.shape)

    return dataSet, lableSet, featSet_approx


def Plot(data, label):
    # 准备绘制箱线图的数据
    boxplot_data = []
    for i in range(len(ecgClassSet)):
        class_data = [data[j] for j in range(len(label)) if label[j] == i]
        boxplot_data.append(class_data)

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, labels=ecgClassSet)
    plt.xlabel('ECG Class')
    plt.ylabel('Value')
    plt.title('Value Boxplot for Each ECG Class')
    plt.grid(True)
    plt.show()


def main():
    # X_train, y_train 是训练集
    # X_test, y_test 是测试集
    # featSet 是特征集合
    X_train, Y_train, featSet = load_data()


if __name__ == '__main__':
    main()
