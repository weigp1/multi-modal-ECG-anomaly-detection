# 定义双流神经网络
from keras import layers, models


def two_stream_model(input_shape):
    # Bidirectional(batch_size, timesteps, input_features)
    # 原始流模型
    raw_input = layers.Input(shape=input_shape, name='raw_input')
    raw_stream = layers.Bidirectional(layers.LSTM(128))(raw_input)
    raw_stream = layers.Dense(64, activation='relu')(raw_stream)
    output_raw = layers.Dense(5, activation='softmax')(raw_stream)

    # 特征流模型
    feat_input = layers.Input(shape=(2 * input_shape[0], input_shape[1]), name='feat_input')
    feat_stream = layers.Bidirectional(layers.LSTM(128))(feat_input)
    feat_stream = layers.Dense(64, activation='relu')(feat_stream)
    output_feat = layers.Dense(5, activation='softmax')(feat_stream)

    # 合并两个流的输出
    output_layer = layers.average([output_raw, output_feat])

    # 创建模型
    model = models.Model(inputs=[raw_input, feat_input], outputs=output_layer)

    return model