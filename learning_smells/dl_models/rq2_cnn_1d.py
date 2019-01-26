from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import configuration, input_data
import inputs
import datetime
import rq1_cnn_1d
import time
# -----------------------------
DIM = "1d"
# TRAINING_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
# EVAL_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_java/"
# OUT_FOLDER = "../results/rq2/raw"
TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_cs/"
EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_java/"
OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"

TRAIN_VALIDATE_RATIO = 0.7
# -----------------------------


def get_all_data():
    print("reading data...")

    # Load training and eval data
    train_data, train_labels, eval_data, eval_labels, max_input_length = \
        inputs.get_data_rq2(training_data_path, eval_data_path, OUT_FOLDER, "rq2_cnn1d_" + smell,
                        train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=5000)

    print("reading data... done.")
    return input_data.Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file():
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    return os.path.join(OUT_FOLDER, "cnn1d_rq2_" + smell + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main():
    input_data = get_all_data()

    conv_layers = {1, 2, 3}
    filters = {8, 16, 32, 64}
    kernels = {5, 7, 11}
    pooling_windows = {2, 3, 4, 5}
    epochs = {50}

    total_iterations = len(conv_layers) * len(filters) * len(kernels) * len(pooling_windows) * len(epochs)
    cur_iter = 1
    outfile = get_out_file()
    write_result(outfile,
                 "conv_layers,filters,kernel,max_pooling_window,epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    for layer in conv_layers:
        for filter in filters:
            for kernel in kernels:
                for pooling_window in pooling_windows:
                    for epoch in epochs:
                        print("** Iteration {0} of {1} **".format(cur_iter, total_iterations))
                        try:
                            config = configuration.CNN_config(layer, filter, kernel, pooling_window, epoch)
                            start_time = time.time()
                            auc, accuracy, precision, recall, f1, average_precision = rq1_cnn_1d.cnn(input_data, config, smell, OUT_FOLDER, DIM)
                            end_time = time.time()
                            elapsed_time = end_time - start_time

                            write_result(outfile, str(layer) + "," + str(filter) + "," + str(kernel) + "," +
                                         str(pooling_window) + "," + str(epoch) + "," +
                                         str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                                         + "," + str(f1) + "," + str(average_precision) + "," +
                                         str(elapsed_time) + "\n")

                        except ValueError as error:
                            print("Skipping combination layer: {}, filter: {}, kernel: {}, pooling_window: {}"
                                  .format(layer, filter, kernel, pooling_window))
                            print(error)
                        cur_iter += 1


if __name__ == "__main__":
    smell_list = {"ComplexMethod", "EmptyCatchBlock", "MagicNumber", "MultifacetedAbstraction"}
    # smell_list = {"ComplexMethod"}
    for smell in smell_list:
        training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
        eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
        inputs.preprocess_data(training_data_path)
        inputs.preprocess_data(eval_data_path)
        main()
