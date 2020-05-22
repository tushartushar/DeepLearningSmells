from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import configuration
import input_data
import inputs
import datetime
import gc
import rq1_rnn_emb_lstm
import time

# --------------------------
DIM = "1d"

MODE = "train_cs"  # "train_cs" or "train_java" representing whether the training data is coming from C# samples or java.

if MODE == "train_cs":
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_java/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_java/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"
else:
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/java_500/tokenizer_out/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/cs_100/tokenizer_out/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/java_500/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/cs_100/tokenizer_out/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"

TRAIN_VALIDATE_RATIO = 0.7


# --------------------------


def get_all_data(training_data_path, eval_data_path, smell):
    print("reading data...")

    if smell in ["ComplexConditional", "ComplexMethod"]:
        max_eval_samples = 150000  # for impl smells (methods)
    else:
        max_eval_samples = 50000  # for design smells (classes)

    training_data, training_labels, eval_data, eval_labels, max_input_length = \
        inputs.get_data_rq2(training_data_path, eval_data_path, OUT_FOLDER, "rq2_rnn_" + smell,
                            train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=5000,
                            max_eval_samples=max_eval_samples)

    training_data = training_data.reshape((len(training_labels), max_input_length))
    eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    print("reading data... done.")
    return input_data.Input_data(training_data, training_labels, eval_data, eval_labels, max_input_length)


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file(smell):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    return os.path.join(OUT_FOLDER, "rnn_rq2_" + smell + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main(training_data_path, eval_data_path, smell, skip_iter=-1):
    data = get_all_data(training_data_path, eval_data_path, smell)
    rnn_layers = {1, 2}
    emb_outputs = [16, 32]
    lstms_units = [32, 64, 128]
    epochs = [50]
    dropouts = [0.2]

    total_iterations = len(emb_outputs) * len(lstms_units) * len(rnn_layers) * len(epochs) * len(dropouts)
    cur_iter = 1
    outfile = get_out_file(smell)
    write_result(outfile,
                 "embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    for layer in rnn_layers:
        for emb_output in emb_outputs:
            for lstm_units in lstms_units:
                for epoch in epochs:
                    for dropout in dropouts:
                        print("** Iteration {0} of {1} **".format(cur_iter, total_iterations))
                        if cur_iter <= skip_iter:
                            cur_iter += 1
                            continue
                        config = configuration.RNN_emb_lstm_config(
                            emb_output=emb_output,
                            lstm_units=lstm_units,
                            layers=layer,
                            epochs=epoch,
                            dropout=dropout)
                        try:
                            start_time = time.time()
                            auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = rq1_rnn_emb_lstm.embedding_lstm(
                                data, config, smell, OUT_FOLDER, iteration=cur_iter)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            result_str = (str(emb_output) + ","
                                          + str(layer) + ","
                                          + str(lstm_units) + ","
                                          + str(epoch) + "," + str(stopped_epoch) + "," +
                                          str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                                          + "," + str(f1) + "," + str(average_precision) + "," +
                                          str(elapsed_time) + "\n")

                            write_result(outfile, result_str)
                        except Exception as ex:
                            print("Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}"
                                  .format(layer, emb_output, lstm_units, epoch))
                            print(ex)
                            write_result(outfile, str(emb_output) + ","
                                         + str(layer) + ","
                                         + str(lstm_units) + ","
                                         + str(epoch) + ",-1,-1,-1,-1,-1,-1,-1\n")
                        cur_iter += 1
                        gc.collect()


def run_rnn_with_best_params(data, smell, emb_output, lstm_units, layer, epoch):
    outfile = get_out_file(smell + "final")
    write_result(outfile,
                 "embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    config = configuration.RNN_emb_lstm_config(
        emb_output=emb_output,
        lstm_units=lstm_units,
        layers=layer,
        epochs=epoch,
        dropout=0.2)
    try:
        start_time = time.time()
        auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = \
            rq1_rnn_emb_lstm.embedding_lstm(data, config, smell, OUT_FOLDER, is_final=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        result_str = (str(emb_output) + ","
                      + str(layer) + ","
                      + str(lstm_units) + ","
                      + str(epoch) + "," + str(stopped_epoch) + "," +
                      str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                      + "," + str(f1) + "," + str(average_precision) + "," +
                      str(elapsed_time) + "\n")

        write_result(outfile, result_str)
    except Exception as ex:
        print("Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}"
              .format(layer, emb_output, lstm_units, epoch))
        print(ex)
        write_result(outfile, str(emb_output) + ","
                     + str(layer) + ","
                     + str(lstm_units) + ","
                     + str(epoch) + ",-1,-1,-1,-1,-1,-1,-1\n")
    gc.collect()


if __name__ == "__main__":
    # smell_list = {"ComplexMethod", "EmptyCatchBlock", "MagicNumber", "MultifacetedAbstraction"}
    # # smell_list = {"ComplexMethod"}
    # for smell in smell_list:
    #     training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    #     eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    #     inputs.preprocess_data(training_data_path)
    #     inputs.preprocess_data(eval_data_path)
    #     main(training_data_path, eval_data_path, smell)

    # # The following is the last step to get the final results. hyper parameters seletected from the best performance (f1)
    smell = "ComplexMethod"
    training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    data1 = get_all_data(training_data_path, eval_data_path, smell)
    run_rnn_with_best_params(data1, smell, emb_output=16, lstm_units=32, layer=1, epoch=5)

    smell = "ComplexConditional"
    training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    data2 = get_all_data(training_data_path, eval_data_path, smell)
    run_rnn_with_best_params(data2, smell, emb_output=32, lstm_units=32, layer=1, epoch=4)

    smell = "FeatureEnvy"
    training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    data3 = get_all_data(training_data_path, eval_data_path, smell)
    run_rnn_with_best_params(data3, smell, emb_output=32, lstm_units=128, layer=1, epoch=10)

    smell = "MultifacetedAbstraction"
    training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
    eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
    data4 = get_all_data(training_data_path, eval_data_path, smell)
    run_rnn_with_best_params(data4, smell, emb_output=32, lstm_units=128, layer=1, epoch=9)
