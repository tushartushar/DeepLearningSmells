from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import configuration, input_data
import inputs
import rq1_cnn_2d
import time

# --- Parameters ---
DIM = "2d"

MODE = "train_cs" # "train_cs" or "train_java" representing whether the training data is coming from C# samples or java.

if MODE == "train_cs":
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_java/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_java/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/program/results/rq2/raw"
else:
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/java_500/tokenizer_out/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/cs_100/tokenizer_out/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/java_500/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/cs_100/tokenizer_out/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/program/results/rq2/raw"

TRAIN_VALIDATE_RATIO = 0.7
# ------------


def get_all_data(training_data_path, eval_data_path, smell):
    print("reading data...")
    train_data, train_labels, eval_data, eval_labels, max_input_height, max_input_width = \
        inputs.get_data_2d_rq2(training_data_path, eval_data_path, OUT_FOLDER, "rq2_cnn2d_" + smell,
                               train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=5000)
    print("reading data... done.")
    return input_data.Input_data2(train_data, train_labels, eval_data, eval_labels,
                                  max_input_height, max_input_width)


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file(smell):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    return os.path.join(OUT_FOLDER, "cnn2d_rq2_" + smell +
                        "_" + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main(training_data_path, eval_data_path, smell, skip_iter=-1):
    input_data = get_all_data(training_data_path, eval_data_path, smell)

    conv_layers = {1, 2, 3}
    filters = {8, 16, 32, 64}
    kernels = {5, 7, 11}
    pooling_windows = {2, 3, 4, 5}
    epochs = {50}

    total_iterations = len(conv_layers) * len(filters) * len(kernels) * len(pooling_windows) * len(epochs)
    cur_iter = 1
    outfile = get_out_file(smell)
    write_result(outfile,
                 "conv_layers,filters,kernel,max_pooling_window,epoch,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    for layer in conv_layers:
        for filter in filters:
            for kernel in kernels:
                for pooling_window in pooling_windows:
                    for epoch in epochs:
                        print("** Iteration {0} of {1} **".format(cur_iter, total_iterations))
                        if cur_iter < skip_iter:
                            cur_iter += 1
                            continue
                        try:
                            config = configuration.CNN_config(layer, filter, kernel, pooling_window, epoch)
                            start_time = time.time()
                            auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = rq1_cnn_2d.cnn(input_data, config, smell, out_folder=OUT_FOLDER, dim=DIM)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            write_result(outfile, str(layer) + "," + str(filter) + "," + str(kernel) + "," +
                                         str(pooling_window) + "," + str(epoch) + "," + str(stopped_epoch) + "," +
                                         str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                                         + "," + str(f1) + "," + str(average_precision) + "," +
                                         str(elapsed_time) + "\n")
                        except ValueError as error:
                            print("Skipping combination layer: {}, filter: {}, kernel: {}, pooling_window: {}"
                                  .format(layer, filter, kernel, pooling_window))
                            print(error)
                            write_result(outfile, str(layer) + "," + str(filter) + "," + str(kernel) + "," +
                                         str(pooling_window) + "," + str(epoch) + ",-1,-1,-1,-1,-1,-1,-1,-1\n")
                        cur_iter += 1

def run_cnn_with_best_params(input_data, layer, filter, kernel, pooling_window, epoch):
    outfile = get_out_file(smell + "final")
    write_result(outfile,
                 "conv_layers,filters,kernel,max_pooling_window,epoch,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    try:
        config = configuration.CNN_config(layer, filter, kernel, pooling_window, epoch)
        start_time = time.time()
        auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = rq1_cnn_2d.cnn(
            input_data, config, smell, out_folder=OUT_FOLDER, dim=DIM, is_final=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        write_result(outfile, str(layer) + "," + str(filter) + "," + str(kernel) + "," +
                     str(pooling_window) + "," + str(epoch) + "," + str(stopped_epoch) + "," +
                     str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                     + "," + str(f1) + "," + str(average_precision) + "," +
                     str(elapsed_time) + "\n")
    except ValueError as error:
        print("Skipping combination layer: {}, filter: {}, kernel: {}, pooling_window: {}"
              .format(layer, filter, kernel, pooling_window))
        print(error)
        write_result(outfile, str(layer) + "," + str(filter) + "," + str(kernel) + "," +
                     str(pooling_window) + "," + str(epoch) + ",-1,-1,-1,-1,-1,-1,-1,-1\n")


if __name__ == "__main__":
    # smell_list = {"ComplexMethod"}
    smell_list = ["ComplexConditional", "ComplexMethod", "MultifacetedAbstraction", "FeatureEnvy"]
    #
    for smell in smell_list:
        training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
        eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
        inputs.preprocess_data_2d(training_data_path)
        inputs.preprocess_data_2d(eval_data_path)
        main(training_data_path, eval_data_path, smell)

    # smell = "ComplexMethod"
    # training_data_path1 = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    # eval_data_path1 = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    # input_data1 = get_all_data(training_data_path1, eval_data_path1, smell)
    # run_cnn_with_best_params(input_data=input_data1, layer=3, filter=16, kernel=7, pooling_window=5, epoch=9)
    #
    # smell = "EmptyCatchBlock"
    # training_data_path2 = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    # eval_data_path2 = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    # input_data2 = get_all_data(training_data_path2, eval_data_path2, smell)
    # run_cnn_with_best_params(input_data=input_data2, layer=3, filter=16, kernel=5, pooling_window=2, epoch=7)
    #
    # smell = "MagicNumber"
    # training_data_path3 = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    # eval_data_path3 = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    # input_data3 = get_all_data(training_data_path3, eval_data_path3, smell)
    # run_cnn_with_best_params(input_data=input_data3, layer=3, filter=16, kernel=7, pooling_window=2, epoch=7)
    #
    # smell = "MultifacetedAbstraction"
    # training_data_path4 = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    # eval_data_path4 = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    # input_data4 = get_all_data(training_data_path4, eval_data_path4, smell)
    # run_cnn_with_best_params(input_data=input_data4, layer=2, filter=16, kernel=7, pooling_window=2, epoch=19)
