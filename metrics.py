import torch
"""
input: pos_score (torch.tensor): 0 dim tensor of probabilities corresponding to positive edges
       neg_score (torch.tensor): 0 dim tensor of probabilities corresponding to negative edges
            """
def mcc(pos_score,neg_score):
    TP = (pos_score >= 0.5).sum() / len(pos_score)
    TN = (neg_score < 0.5).sum() / len(neg_score)
    FP = 1 - TN
    FN = 1 - TP
    if TP * TN - FP * FN==0:
        mcc = 0
    else:
        mcc = (TP * TN - FP * FN) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return mcc


def balanced_acc(pos_score,neg_score):
    TP = (pos_score >= 0.5).sum() / len(pos_score)
    TN = (neg_score < 0.5).sum() / len(neg_score)

    acc=(TP+TN)/2

    return acc
