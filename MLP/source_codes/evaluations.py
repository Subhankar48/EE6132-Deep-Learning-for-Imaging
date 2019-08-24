import numpy as np

def accuracy(y_pred, y_true):
    total_coulmns = np.shape(y_true)[1]
    matching_colums = (np.argmax(y_pred, axis=0)==np.argmax(y_true, axis=0)).sum()
    return matching_colums/total_coulmns

def precision(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_pred))

def recall(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_true))

def f1_score(y_pred, y_true):
    _precision = precision(y_pred,y_true)
    _recall = recall(y_pred, y_true)
    return 2*_precision*_recall/(_precision+_recall)
    
