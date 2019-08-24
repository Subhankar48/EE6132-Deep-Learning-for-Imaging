import numpy as np

def precision(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_pred))

def recall(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_true))

def f1_score(y_pred, y_true):
    _precision = precision(y_pred,y_true)
    _recall = recall(y_pred, y_true)
    return 2*_precision*_recall/(_precision+_recall)
    