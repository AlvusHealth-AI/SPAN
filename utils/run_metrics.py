import os
import os.path as osp

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

np.random.seed(333)  # For reproducibility
np.set_printoptions(precision=2)

for filename in os.listdir("."):
    if ".csv" not in filename:
        continue
    print(filename)

    data = pd.read_csv(filename)
    accs, aurocs, auprcs, pres, recs, f1s, baccs, mccs = [], [], [], [], [], [], [], []
    for i in range(5):
        test_set = data[data["Fold"] == i]
        gt = test_set["Disruption"]
        pred = test_set["PredDisruption"]
        prob = test_set["Prob"]
        acc = accuracy_score(gt, pred)
        auroc = roc_auc_score(gt, prob)
        auprc = average_precision_score(gt, prob)
        pre = precision_score(gt, pred)
        rec = recall_score(gt, pred)
        f1 = f1_score(gt, pred)
        bacc = balanced_accuracy_score(gt, pred)
        mcc = matthews_corrcoef(gt, pred)

        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)
        pres.append(pre)
        recs.append(rec)
        f1s.append(f1)
        baccs.append(bacc)
        mccs.append(mcc)

    print(
        f"Accuracy: {np.median(accs):.2f} ({np.percentile(accs, 25):.2f}, {np.percentile(accs, 75):.2f})"
    )
    print(
        f"AUROC: {np.median(aurocs):.2f} ({np.percentile(aurocs, 25):.2f}, {np.percentile(aurocs, 75):.2f})"
    )
    print(
        f"AUPRC: {np.median(auprcs):.2f} ({np.percentile(auprcs, 25):.2f}, {np.percentile(auprcs, 75):.2f})"
    )
    print(
        f"Precision: {np.median(pres):.2f} ({np.percentile(pres, 25):.2f}, {np.percentile(pres, 75):.2f})"
    )
    print(
        f"Recall: {np.median(recs):.2f} ({np.percentile(recs, 25):.2f}, {np.percentile(recs, 75):.2f})"
    )
    print(
        f"F1: {np.median(f1s):.2f} ({np.percentile(f1s, 25):.2f}, {np.percentile(f1s, 75):.2f})"
    )
    print(
        f"Balanced Accuracy: {np.median(baccs):.2f} ({np.percentile(baccs, 25):.2f}, {np.percentile(baccs, 75):.2f})"
    )
    print(
        f"MCC: {np.median(mccs):.2f} ({np.percentile(mccs, 25):.2f}, {np.percentile(mccs, 75):.2f})"
    )
    print("\n")
