import numpy as np
from sklearn.metrics import recall_score, precision_score

def find_best_threshold(y_test, y_prob):
    thresholds = np.arange(0.2, 0.51, 0.05)

    for t in thresholds:
        y_pred_new = (y_prob >= t).astype(int)
        print(f"Threshold = {t:.2f}")
        print(" Recall:", recall_score(y_test, y_pred_new))
        print(" Precision:", precision_score(y_test, y_pred_new))
        print("-" * 40)

    # Choose manually or automate
    return 0.35
