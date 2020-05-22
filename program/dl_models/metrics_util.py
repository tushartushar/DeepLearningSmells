from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def get_all_metrics(model, eval_data, eval_labels, pred_labels):
    fpr, tpr, thresholds_keras = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print("auc_keras:" + str(auc_))

    score = model.evaluate(eval_data, eval_labels, verbose=0)
    print("Test accuracy: " + str(score[1]))

    precision = precision_score(eval_labels, pred_labels)
    print('Precision score: {0:0.2f}'.format(precision))

    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {0:0.2f}'.format(recall))

    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {0:0.2f}'.format(f1))

    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    return auc_, score[1], precision, recall, f1, average_precision, fpr, tpr


def get_all_metrics_(eval_labels, pred_labels):
    fpr, tpr, thresholds_keras = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print("auc_keras:" + str(auc_))

    precision = precision_score(eval_labels, pred_labels)
    print('Precision score: {0:0.2f}'.format(precision))

    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {0:0.2f}'.format(recall))

    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {0:0.2f}'.format(f1))

    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    return auc_, precision, recall, f1, average_precision, fpr, tpr