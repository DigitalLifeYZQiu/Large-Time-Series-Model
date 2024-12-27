import numpy as np
from utils.tools import adjustment
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tqdm import tqdm,trange

def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
            for j in range(i, len(score)):
                if actual[j] == 0:
                    break
                else:
                    if predict[j] == 0:
                        predict[j] = 1

        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    
def adjbestf1(y_true: np.array, y_scores: np.array, n_splits: int = 50):
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    
    # thresholds = np.linspace(np.percentile(y_scores,95), y_scores.max(), 10)
    adjusted_f1,f1 = [], []

    for threshold in thresholds:
        y_pred = y_scores >= threshold
        score_real = f1_score(y_pred, y_true)
        # score_real = precision_recall_fscore_support(y_true, y_pred, average='binary')[2]
        f1.append([score_real, threshold])
        # import pdb
        # pdb.set_trace()
        # 将预测结果进行adjust操作
        y_pred_ajust = adjust_predicts(
            score=y_scores,
            label=(y_true > 0),
            pred=y_pred,
            threshold=None,
            calc_latency=False,
        )
        # import pdb
        # pdb.set_trace()
        # y_pred_ajust = adjustment(y_true, y_scores)[1]
        # ! Changed here
        # score_real = f1_score(y_pred, y_true)
        # score_real = precision_recall_fscore_support(y_true, y_pred, average='binary')[2]
        # f1.append([score_real, threshold])
        # ! End of Change

        score_ajust = f1_score(y_pred_ajust, y_true)
        adjusted_f1.append([score_ajust, threshold])

        # score = f1_score(y_pred, y_true)# 获得对应阈值下的f1值
        # print("counting adjustment best f1  shreshold:{:0.4f}_score:{:0.4f}".format(threshold,score_ajust))
        # print("counting adjustment f1  shreshold:{:0.4f}_score:{:0.4f}".format(threshold,score))

    adjusted_f1 = np.array(adjusted_f1)
    f1 = np.array(f1)
    # import pdb
    # pdb.set_trace()
    best_adjusted_f1_index = np.argmax(adjusted_f1[:, 0])
    best_f1_index = np.argmax(f1[:, 0])
    y_best_pred_adj = y_scores >= adjusted_f1[best_adjusted_f1_index][1]
    y_best_pred = y_scores >= f1[best_f1_index][1]

    return y_best_pred_adj, y_best_pred