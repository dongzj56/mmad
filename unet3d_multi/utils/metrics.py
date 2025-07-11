import os, json, time, csv, numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, auc)

# ----------------- 指标函数 -------------------
def calculate_metrics(y_true, y_pred, y_score):
    if len(y_true) == 0:
        raise ValueError("No samples to evaluate. Please check your test_loader / data split.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp + 1e-8)
    return {
        'ACC': accuracy_score(y_true, y_pred),
        'PRE': precision_score(y_true, y_pred, zero_division=0),
        'SEN': recall_score(y_true, y_pred, zero_division=0),
        'SPE': spe,
        'F1' : f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'cm' : np.array([[tn, fp], [fn, tp]])
    }