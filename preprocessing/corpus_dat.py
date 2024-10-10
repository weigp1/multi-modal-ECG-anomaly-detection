import os
import pandas as pd
import wfdb
import pywt
import numpy as np
from scipy.stats import kurtosis, skew as skews
from scipy import signal

exp_path = 'D:/myFile/'
dataSets = ['mitdb']
frequency = [360]
labelSet = ['N', 'A', 'V', 'S']
classSet = ['Normal', 'Atrial', 'Ventricular', 'Supra-ventricular']
description = ['low', 'medium low', 'medium', 'medium high', 'high']


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


if __name__ == '__main__':
    scanned = []
    for index in range(len(dataSets)):
        f = frequency[index]
        dataset_path = os.path.join(exp_path, dataSets[index] + '/')
        files = os.listdir(dataset_path)
        data_stats = []

        kurtosis_list = []
        skewness_list = []
        rr_intervals = []

        for file in files:
            name = os.path.splitext(file)[0]
            if name in scanned:
                continue
            scanned.append(name)
            print(name)
            try:
                annotation = wfdb.rdann(os.path.join(dataset_path, name), 'atr')
                rdata = wfdb.rdrecord(os.path.join(dataset_path, name), channels=[1]).p_signal
                rdata = denoise(rdata.flatten())

                samples = annotation.sample
                symbols = annotation.symbol

                prev_sample = samples[0]
                for i in range(1, len(samples)):
                    if symbols[i] not in labelSet:
                        continue
                    RR_interval = samples[i] - prev_sample
                    prev_sample = samples[i]
                    if RR_interval <= 100 or RR_interval >= 360:  # Skip non-positive intervals
                        continue
                    if samples[i] - f / 2 <= 0:
                        continue
                    elif samples[i] + f / 2 > len(rdata):
                        break

                    segment = rdata[samples[i] - int(f / 2):samples[i] + int(f / 2)]

                    segment = signal.resample(segment, 250)

                    Kurt = np.around(np.array(kurtosis(segment)), decimals=2)
                    Skew = np.around(np.array(skews(segment)), decimals=2)

                    kurtosis_list.append(Kurt)
                    skewness_list.append(Skew)
                    rr_intervals.append(RR_interval)

                    p_wave_index = np.argmax(segment[0:125 - 5])
                    q_wave_index = 125 - 25 + np.argmin(segment[100:125])
                    s_wave_index = 125 + np.argmin(segment[125:150])
                    t_wave_index = 125 + 5 + np.argmax(segment[125 + 5:250])

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
                    
                    index = labelSet.index(symbols[i])
                    data_stats.append((feature_summary, classSet[index]))

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

        # 创建数据表并保存为 CSV 文件
        df_stats = pd.DataFrame(data_stats, columns=['text', 'label'])
        df_stats.to_csv(os.path.join('corpus_' + dataSets[index] + '.csv'), index=False)

        # 打印每个特征的分布范围和中位值
        if kurtosis_list:
            print(f"Kurtosis range: {min(kurtosis_list)} to {max(kurtosis_list)}")
            print(f"Kurtosis median: {np.median(kurtosis_list)}")
        if skewness_list:
            print(f"Skewness range: {min(skewness_list)} to {max(skewness_list)}")
            print(f"Skewness median: {np.median(skewness_list)}")
        if rr_intervals:
            print(f"RR Interval range: {min(rr_intervals)} to {max(rr_intervals)}")
            print(f"RR Interval median: {np.median(rr_intervals)}")
