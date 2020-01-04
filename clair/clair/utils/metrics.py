from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
import numpy as np

def pct_error(y_real, y_pred):
    return abs((y_pred - y_real)/ y_real)


def f1_score(y_true, y_pred):
    if (y_true != 1).all():
        return 0

    return f1(y_true, y_pred)


def precision_score(data, response, model):
    y_true = np.sign((response - response.shift(1))[1:]).replace(0, -1)
    y_pred = np.sign((model.predict(data) - response.shift(1))[1:]).replace(0, -1)
    return precision(y_true, y_pred)
