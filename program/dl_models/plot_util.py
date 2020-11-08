from inspect import signature

import matplotlib.pyplot as plt
import datetime
import os
import configuration as cfg
# from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve

def save_roc_curve(fpr, tpr, roc_auc, smell, config, out_folder, dim):
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    file_name = get_plot_file_name(smell, config, out_folder, dim, "_roc_")
    fig.savefig(file_name)

def save_precision_recall_curve(eval_labels, pred_labels, average_precision, smell, config, out_folder, dim, method):
    fig = plt.figure()
    precision, recall, _ = precision_recall_curve(eval_labels, pred_labels)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if isinstance(config, cfg.CNN_config):
        title_str = smell + " (" + method + " - " + dim + ") - L=" + str(config.layers) + ", E=" + str(config.epochs) + ", F=" + str(config.filters) + \
                    ", K=" + str(config.kernel) + ", PW=" + str(config.pooling_window) + ", AP={0:0.2f}".format(average_precision)
    # plt.title(title_str)
    # plt.show()
    file_name = get_plot_file_name(smell, config, out_folder, dim, method, "_prc_")
    fig.savefig(file_name)

def get_plot_file_name(smell, config, out_folder, dim, method, plot_type = "_prc_"):
    now = datetime.datetime.now()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if isinstance(config, cfg.RNN_emb_lstm_config):
        return os.path.join(out_folder,
                            smell + "_" + method + "_" + plot_type + str(config.layers) + "_" + str(config.lstm_units) + "_"
                            + str(config.emb_output) + "_" + str(config.dropout) + "_" + str(config.epochs) + "_"
                            + str(now.strftime("%d%m%Y_%H%M") + ".pdf"))
    if isinstance(config, cfg.CNN_config):
        return os.path.join(out_folder, smell + "_" + method + dim + plot_type + str(config.layers) + "_" + str(config.filters) + "_"
        + str(config.kernel) + "_" + str(config.pooling_window) + "_" + str(config.epochs) + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".pdf"))
