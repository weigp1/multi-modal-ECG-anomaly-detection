import os
import wfdb
import pywt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew as skews
from scipy import signal

from HFD import HFD

# 随机种子
RANDOM_SEED = 42
# 数据标签
ecgClassSet = ['N', 'A', 'V', 'L', 'R']

# 1 表示使用原始数据导入，2表示使用csv数据导入
load_type = 1

# 原始数据导入参数
raw_path = 'E:\\ECG_Dataset\\'
dataSets = ['mitdb']
frequency = [360]


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
    coeffs = pywt.swt(data, 'db2', level=1, norm=True)

    # 计算小波方差
    wavelet_var = []
    for coeff in coeffs:
        variances = [(w ** 2).mean() for w in coeff]
        wavelet_var.extend(variances)

    return coeffs[-1], wavelet_var


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
    Skew = skews(data)
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


# PCA 降维
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

    return scores


# 导入数据，进行预处理
def load_data_raw():
    print("load dataset by raw now")
    for index in range(len(dataSets)):
        dataSet = []
        featSet = []
        labelSet = []
        corpus = []
        print(dataSets[index])

        f = frequency[index]
        file_names_set = set()

        for file in os.listdir(raw_path + dataSets[index] + '/'):
            name = os.path.splitext(file)[0]

            # 防止重复读取
            if name in file_names_set:
                continue
            file_names_set.add(name)

            annotation = wfdb.rdann(raw_path + dataSets[index] + '/' + name, 'atr')
            samples = annotation.sample
            symbol = annotation.symbol

            rdata = wfdb.rdrecord(raw_path + dataSets[index] + '/' + name, channels=[0]).p_signal
            rdata = rdata.flatten()
            data = denoise(rdata)

            prev_sample = samples[0]
            i = 1
            j = len(symbol) - 1

            while i < j:
                i += 1
                if samples[i] - f / 2 <= 0:
                    continue
                elif samples[i] + f / 2 > len(data):
                    break
                if symbol[i] not in ecgClassSet:
                    continue
                # 加入数据和标签
                x_data = data[int(samples[i] - f / 2):int(samples[i] + f / 2)]
                x_data = signal.resample(x_data, 250)

                labelSet.append(ecgClassSet.index(symbol[i]))

                dataSet.append(x_data)

                feat_sf = np.array(SF(x_data))
                feat_ar = np.array(ARC(x_data))
                feat_se = np.array(SE(x_data))
                wave, wv = WV(x_data)
                feat_wv = np.array(wv)
                feat_wav = np.array(wave)
                feat_hfd = np.array(HFD(x_data))
                featSet.append(np.concatenate((feat_sf, feat_ar, feat_se, feat_wv, feat_hfd, feat_wav), axis=None))

                RR_interval = samples[i] - prev_sample
                prev_sample = samples[i]
                Kurt = np.around(np.array(kurtosis(x_data)), decimals=2)
                Skew = np.around(np.array(skews(x_data)), decimals=2)

                p_wave_index = np.argmax(x_data[0:125 - 5])
                q_wave_index = 125 - 25 + np.argmin(x_data[100:125])
                s_wave_index = 125 + np.argmin(x_data[125:150])
                t_wave_index = 125 + 5 + np.argmax(x_data[125 + 5:250])

                description = ['low', 'medium low', 'medium', 'medium high', 'high']

                if Kurt <= 0:
                    kurt = description[0]
                elif 0 < Kurt <= 10:
                    kurt = description[1]
                elif 10 < Kurt <= 20:
                    kurt = description[2]
                elif 20 < Kurt <= 30:
                    kurt = description[3]
                else:
                    kurt = description[4]

                if Skew <= -1:
                    skew = description[0]
                elif -1 < Skew <= 1:
                    skew = description[1]
                elif 1 < Skew <= 3:
                    skew = description[2]
                elif 3 < Skew <= 5:
                    skew = description[3]
                else:
                    skew = description[4]

                if RR_interval <= 150:
                    rr_interval = description[0]
                elif 150 < RR_interval <= 200:
                    rr_interval = description[1]
                elif 200 < RR_interval <= 300:
                    rr_interval = description[2]
                elif 300 < RR_interval <= 350:
                    rr_interval = description[3]
                else:
                    rr_interval = description[4]

                feature_summary = (
                        "This ECG wave has a " + kurt + " kurtosis, a " + skew + " skewness, a " + rr_interval + " R-peak interval,"
                                                                                                                 f"with P peak at timestamp {p_wave_index}, "
                                                                                                                 f"Q peak at timestamp {q_wave_index}, "
                                                                                                                 f"S peak at timestamp {s_wave_index}, "
                                                                                                                 f"and T peak at timestamp {t_wave_index}.")
                corpus.append(feature_summary)

        # 重排列 & 分割数据集
        df = pd.DataFrame(dataSet)
        df.to_csv('../dataset/dataSet_' + dataSets[index] + '.csv', index=False)

        df = pd.DataFrame(labelSet)
        df.to_csv('../dataset/labelSet_' + dataSets[index] + '.csv', index=False)

        featSet = np.array(featSet)
        df = pd.DataFrame(featSet)
        df.to_csv('../dataset/featSet_' + dataSets[index] + '.csv', index=False)

        corpus = np.array(corpus)
        df = pd.DataFrame(corpus, columns=["text"])
        df.to_csv('../dataset/corpus_' + dataSets[index] + '.csv', index=False)

        # SVD奇异值分解
        # print('Before SVD: ', featSet.shape)
        # scores = pca_with_svd(featSet, 15)
        # print('After SVD: ', scores.shape)
        # df = pd.DataFrame(scores)


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
    if load_type == 1:
        load_data_raw()
    # else:
    #     load_data_csv()
