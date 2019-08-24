import numpy as np

def accuracy(y_pred, y_true):
    total_coulmns = np.shape(y_true)[1]
    correct_columns = 0
    for index in range(total_coulmns):
        if (all(y_pred[:,index] == y_true[:,index])):
            correct_columns = correct_columns+1
    return correct_columns/total_coulmns

def precision(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_pred))

def recall(y_pred, y_true):
    np.sum(np.sum(y_pred*y_true))/np.sum(np.sum(y_true))

def f1_score(y_pred, y_true):
    _precision = precision(y_pred,y_true)
    _recall = recall(y_pred, y_true)
    return 2*_precision*_recall/(_precision+_recall)
    
