from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np
from scipy.interpolate import interp1d

def score(idd, ood, order=0, metric=0):
    # computes (0) AUROC, (1) AUPRC, or (2) FPR@80%TPR for given id and ood combination
    
    idd = np.squeeze(idd)[:5000]
    ood = np.squeeze(ood)[:5000]
    if order == 0:
        y_true = np.concatenate((np.ones_like(idd), np.zeros_like(ood)), axis=0)
    elif order == 1:
        y_true = np.concatenate((np.zeros_like(idd), np.ones_like(ood)), axis=0)
    y_pred = np.concatenate((idd, ood), axis=0)
    if metric == 0:
        ans = roc_auc_score(y_true, y_pred)
    elif metric == 1:
        ans = average_precision_score(y_true, y_pred)
    elif metric == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
        ind = np.argmax(tpr>0.8)  
        x = np.array((tpr[ind-1], tpr[ind]))
        y = np.array((fpr[ind-1], fpr[ind]))    
        f = interp1d(x,y)
        ans = f(0.8)
    return ans