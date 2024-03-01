import numpy as np
from sklearn import linear_model


# feature extraction of two features
def handcrafted_feat(data):
    fea = np.zeros((data.shape[0] - 1, 2 * data.shape[1]))
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[0] - 1):
        for j in range(data.shape[1]):
            fea[i, 2 * j] = np.mean(data[:(i + 2), j])
            regr = linear_model.LinearRegression()
            regr.fit(x[:(i + 2)].reshape(-1, 1), np.ravel(data[:(i + 2), j]))
            fea[i, 2 * j + 1] = regr.coef_

    return fea
