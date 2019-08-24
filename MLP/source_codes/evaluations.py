import numpy as np

def accuracy(y_pred, y_true):
    total_coulmns = np.shape(y_true)[1]
    matching_colums = ((np.argmax(y_pred, axis=0)==np.argmax(y_true, axis=0))*np.ones(total_coulmns)).sum()
    return matching_colums/total_coulmns

def precision(y_pred, y_true):
    (y_pred*y_true).sum()/(y_pred).sum()

def recall(y_pred, y_true):
    (y_pred*y_true).sum()/(y_true).sum()

def f1_score(y_pred, y_true):
    _precision = precision(y_pred,y_true)
    _recall = recall(y_pred, y_true)
    return 2*_precision*_recall/(_precision+_recall)
    
def confusion_matrix(y_pred, y_true):
    confusion_mat =  np.dot(y_true, np.transpose(y_pred))
    total_samples = np.sum(np.sum(confusion_mat))
    correct_samples = np.trace(confusion_mat)
    no_of_samples_model_underpredicted = np.tril(confusion_mat,-1).sum()
    no_of_samples_model_overpredicted = np.triu(confusion_mat,1).sum()
    return confusion_mat, no_of_samples_model_underpredicted, no_of_samples_model_overpredicted