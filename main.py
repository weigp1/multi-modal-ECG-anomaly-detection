import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from model import two_stream_model
from preprocessing.loading import load_data

# 项目根目录
project_path = "../"

BATCH_SIZE = 128
NUM_EPOCHS = 10


def calculate_metrics(y_true, y_pred):
    # 计算Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # 计算Precision
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    # 计算Recall
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    # 计算ROC和AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # 训练集与测试集
    X_train, X_test, X_train_H, X_test_H, y_train, y_test = load_data('mitdb')
    if os.path.exists(model_path):
        # 如果预训练模型存在，则直接导入
        print('导入预训练模型，跳过训练过程')
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # 创建双流神经网络模型
        raw_input_shape = (X_train.shape[1], X_train.shape[2])
        model = two_stream_model(raw_input_shape)

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit([X_train, X_train_H], y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # 评估模型
    model.evaluate([X_test, X_test_H], y_test)

    # 参数评估
    y_pred_probs = model.predict([X_test, X_test_H])
    y_pred = np.argmax(y_pred_probs, axis=1)
    calculate_metrics(y_test, y_pred)