import numpy as np

def accuracy(y_pred, y_true):
    total_coulmns = np.shape(y_true)[1]
    matching_colums = ((np.argmax(y_pred, axis=0)==np.argmax(y_true, axis=0))*np.ones(total_coulmns)).sum()
    return matching_colums/total_coulmns

def confusion_matrix(y_pred, y_true):
    confusion_mat =  np.dot(y_true, np.transpose(y_pred))
    precision = np.diag(confusion_mat)/np.sum(confusion_mat, axis=0)
    recall = np.diag(confusion_mat)/(np.sum(confusion_mat, axis=1))
    f1_score = 2*precision*recall/(precision+recall)
    return confusion_mat, precision, recall, f1_score