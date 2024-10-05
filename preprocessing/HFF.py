import os

import pandas as pd
import wfdb
import pywt
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from hfd import HFD
import statsmodels.api as sm
from matplotlib import pyplot as plt

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
def ARC(data, order=19):
    # 拟合 AR 模型
    model = sm.tsa.AutoReg(data, lags=order)
    ar_model = model.fit()
    # 返回 AR 系数
    return ar_model.params


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


#
def pca_with_svd(data, num_components):
    standardized_data = data

    # 使用SVD计算主成分
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)

    # 计算奇异值的平方并归一化
    normalized_eigenvalues = (S ** 2) / np.sum(S ** 2)

    # 计算保留的信息占比
    retained_variance_ratio = np.sum(normalized_eigenvalues[:num_components])
    print(retained_variance_ratio)

    # 选择主成分
    Vt_reduced = Vt[:num_components, :]

    # 计算主成分得分
    scores = np.dot(standardized_data, Vt_reduced.T)

    # 肘部法则
    # silhouette_scores = []
    # for num_components in range(10, 31):
    #     # 使用SVD计算主成分
    #     U, S, Vt = np.linalg.svd(data, full_matrices=False)
    #     # 选择主成分
    #     Vt_reduced = Vt[:num_components, :]
    #     # 计算主成分得分
    #     scores = np.dot(data, Vt_reduced.T)
    #     kmeans = KMeans(n_clusters=5, random_state=42)
    #     labels = kmeans.fit_predict(scores)
    #     silhouette_scores.append(silhouette_score(scores, labels))
    #
    # # 绘制肘部法则图
    # plt.plot(range(10, 31), silhouette_scores, marker='o')
    # plt.xlabel('Number of Dimensions')
    # plt.ylabel('Silhouette Score')
    # plt.title('Elbow Method for Dimensionality Reduction')
    # plt.show()

    return scores


origin_path = 'D:/myproject/ECG_Benchmark/data/'
filtered_path = '../data/'
dataSets = ['mitdb']
frequency = [360]


# 导入数据，进行预处理
def load_data():
    for index in range(len(dataSets)):
        dataSet = []
        featSet = []
        print(dataSets[index])
        f = frequency[index]
        try:
            for file in os.listdir(filtered_path + dataSets[index] + '/'):
                name = os.path.splitext(file)[0]
                try:
                    annotation = wfdb.rdann(origin_path + dataSets[index] + '/' + name, 'atr')
                    samples = annotation.sample
                    rdata = wfdb.rdrecord(origin_path + dataSets[index] + '/' + name, channels=[0]).p_signal
                    rdata = rdata.flatten()
                    data = denoise(rdata)
                    i = 0
                    j = len(annotation.symbol) - 1

                    while i < j:
                        i += 1
                        if samples[i] - f/2 <= 0:
                            continue
                        elif samples[i] + f/2 > len(data):
                            break
                        x_data = data[int(samples[i] - f/2):int(samples[i] + f/2)]
                        dataSet.append(x_data)

                        feat_sf = np.array(SF(x_data))
                        feat_ar = np.array(ARC(x_data))
                        feat_se = np.array(SE(x_data))
                        feat_wv = np.array(WV(x_data))
                        feat_hfd = np.array(HFD(x_data))
                        featSet.append(np.concatenate((feat_sf, feat_ar, feat_se, feat_wv, feat_hfd), axis=None))
                except Exception:
                    continue
        except Exception:
            continue
        # 重排列 & 分割数据集
        df = pd.DataFrame(dataSet)

        df.to_csv('../benchmark/dataSet_' + dataSets[index] + '.csv', index=False)

        # SVD奇异值分解
        featSet = np.array(featSet)
        print('Before SVD: ', featSet.shape)
        scores = pca_with_svd(featSet, 15)
        print('After SVD: ', scores.shape)
        df = pd.DataFrame(scores)
        df.to_csv('../benchmark/featSet_' + dataSets[index] + '.csv', index=False)


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


if __name__ == '__main__':
    load_data()
