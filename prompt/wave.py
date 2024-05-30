import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

origin_path = 'E:/ECG_Dataset/'  # 原始数据集路径
filtered_path = '../data/'  # 过滤后数据存储路径
dataSets = ['mitdb']  # 要处理的数据集名称列表


def main():
    for index in range(len(dataSets)):
        dataset_path = os.path.join(filtered_path, dataSets[index] + '/')  # 构建过滤后数据集的路径
        origin_dataset_path = os.path.join(origin_path, dataSets[index] + '/')  # 构建原始数据集的路径
        files = os.listdir(dataset_path)  # 获取数据集路径下的文件列表

        for file in files:
            name = os.path.splitext(file)[0]
            annotation = wfdb.rdann(os.path.join(origin_dataset_path, name), 'atr')  # 读取注释数据
            signal = wfdb.rdrecord(os.path.join(origin_dataset_path, name), channels=[0]).p_signal  # 读取ECG信号

            # 获取注释的样本点（R波位置）
            r_peaks = annotation.sample
            num = 0

            for r_peak in r_peaks:
                if num < 2:
                    num += 1
                    continue
                segment = signal[r_peak - 125:r_peak + 125]
                p_wave_index = np.argmax(segment[0:125-5])
                q_wave_index = 125-25+np.argmin(segment[100:125])
                s_wave_index = 125+np.argmin(segment[125:150])
                t_wave_index = 125+5+np.argmax(segment[125+5:250])

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


if __name__ == "__main__":
    main()
