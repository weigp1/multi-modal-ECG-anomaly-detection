import os
import pandas as pd
import wfdb
import pywt
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew
from scipy import signal

origin_path = 'E:/ECG_Dataset/'
filtered_path = '../data/'
dataSets = ['mitdb']
frequency = [360]
classSet = ['N', 'A', 'V', 'L', 'R']


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
    for index in range(len(dataSets)):
        f = frequency[index]
        dataset_path = os.path.join(filtered_path, dataSets[index] + '/')
        origin_dataset_path = os.path.join(origin_path, dataSets[index] + '/')
        files = os.listdir(dataset_path)
        data_stats = []

        for file in files:
            name = os.path.splitext(file)[0]
            try:
                annotation = wfdb.rdann(os.path.join(origin_dataset_path, name), 'atr')
                rdata = wfdb.rdrecord(os.path.join(origin_dataset_path, name), channels=[0]).p_signal
                rdata = denoise(rdata.flatten())

                samples = annotation.sample
                symbols = annotation.symbol

                prev_sample = samples[0]
                for i in range(1, len(samples)):
                    RR_interval = samples[i] - prev_sample
                    prev_sample = samples[i]
                    if RR_interval <= 0:  # Skip non-positive intervals
                        continue
                    if samples[i] - f / 2 <= 0:
                        continue
                    elif samples[i] + f / 2 > len(rdata):
                        break

                    segment = rdata[samples[i] - int(f / 2):samples[i] + int(f / 2)]

                    segment = signal.resample(segment, 250)

                    Kurt = np.array(kurtosis(segment))
                    Skew = np.array(skew(segment))

                    p_wave_index = np.argmax(segment[0:125 - 5])
                    q_wave_index = 125 - 25 + np.argmin(segment[100:125])
                    s_wave_index = 125 + np.argmin(segment[125:150])
                    t_wave_index = 125 + 5 + np.argmax(segment[125 + 5:250])

                    # 画出信号
                    plt.figure(figsize=(10, 6))
                    plt.plot(segment)

                    # 标记P波的位置
                    plt.scatter(p_wave_index, segment[p_wave_index], color='green', label='P wave')

                    # 标记Q波的位置
                    plt.scatter(q_wave_index, segment[q_wave_index], color='blue', label='Q wave')

                    # 标记S波的位置
                    plt.scatter(s_wave_index, segment[s_wave_index], color='blue', label='S wave')

                    # 标记T波的位置
                    plt.scatter(t_wave_index, segment[t_wave_index], color='green', label='T wave')

                    plt.xlabel('Sample')
                    plt.ylabel('Amplitude')
                    plt.title('ECG Signal with Q and S Waves')
                    plt.legend()
                    plt.show()

                    # 将特征总结为一句话
                    feature_summary = f"The signal has Kurtosis {Kurt}, Skewness {Skew}, and RR Interval {RR_interval}"

                    # 标签
                    symbol = symbols[i]

                    data_stats.append((feature_summary, symbol))

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

        # 创建数据表并保存为 CSV 文件
        df_stats = pd.DataFrame(data_stats, columns=['text', 'category'])
        df_stats.to_csv(os.path.join('stats_' + dataSets[index] + '.csv'), index=False)
