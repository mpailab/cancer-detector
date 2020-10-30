from sklearn.metrics import *


def TPR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TP / (TP + FN)


def TNR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TN / (TN + FP)


def min_TPR_TNR(y_true, y_pred):
    return min(TNR(y_true, y_pred), TPR(y_true, y_pred))
