# It is a combined implementation for rnn for both rq1 and rq2
# To save the training time

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import configuration
import input_data
import inputs
import datetime
import numpy as np
import gc
import time
import metrics_util
import plot_util

# --- Parameters --
DIM = "1d"

TRAINING_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
RQ1_OUT_FOLDER = "../results/rq1/raw"
RQ2_OUT_FOLDER = "../results/rq2/raw"
EVAL_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_java/"
# TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_cs/"
# EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_java/"
# RQ1_OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"
# RQ2_OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"

TRAIN_VALIDATE_RATIO = 0.7
CLASSIFIER_THRESHOLD = 0.7
# ---


def embedding_lstm(rq1_data, rq2_data, config, smell, rq1_out_folder=RQ1_OUT_FOLDER, rq2_out_folder = RQ2_OUT_FOLDER, dim = DIM):
    tf.keras.backend.clear_session()
    max_features = int(max(np.max(rq1_data.train_data), np.max(rq1_data.eval_data)))
    print("max features: " + str(max_features))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,
                                        output_dim=config.emb_output,
                                        mask_zero=True))
    for i in range(0, config.layers - 1):
        model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    # model.add(tf.keras.layers.Dropout(config.dropout))
    model.add(tf.keras.layers.LSTM(config.lstm_units, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.Dropout(config.dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0001,
                                                 patience=2,
                                                 verbose=1,
                                                 mode='auto')
    best_model_filepath = 'weights_best.rnn.' + smell + '.hdf5'
    if os.path.exists(best_model_filepath):
        print("deleting the old weights file..")
        os.remove(best_model_filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks_list = [earlystop, checkpoint]

    batch_sizes = [32, 64, 128, 256]
    b_size = int(len(rq1_data.train_labels) / 512)
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1

    model.fit(rq1_data.train_data,
                        rq1_data.train_labels,
                        validation_split=0.2,
                        epochs=config.epochs,
                        batch_size=batch_sizes[b_size],
                        callbacks=callbacks_list)
    # y_pred = model.predict(data.eval_data).ravel()

    stopped_epoch = earlystop.stopped_epoch
    model.load_weights(best_model_filepath)

    # y_pred = model.predict_classes(data.eval_data)
    # We manually apply classification threshold
    rq1_prob = model.predict_proba(rq1_data.eval_data)
    rq1_y_pred = inputs.get_predicted_y(rq1_prob, CLASSIFIER_THRESHOLD)

    rq2_prob = model.predict_proba(rq2_data.eval_data)
    rq2_y_pred = inputs.get_predicted_y(rq2_prob, CLASSIFIER_THRESHOLD)

    rq1_auc, rq1_accuracy, rq1_precision, rq1_recall, rq1_f1, rq1_average_precision, rq1_fpr, rq1_tpr = \
        metrics_util.get_all_metrics(model, rq1_data.eval_data, rq1_data.eval_labels, rq1_y_pred)
    rq2_auc, rq2_accuracy, rq2_precision, rq2_recall, rq2_f1, rq2_average_precision, rq2_fpr, rq2_tpr = \
        metrics_util.get_all_metrics(model, rq2_data.eval_data, rq2_data.eval_labels, rq2_y_pred)

    # plot_util.save_roc_curve(fpr, tpr, auc, smell, config, out_folder, DIM)
    plot_util.save_precision_recall_curve(rq1_data.eval_labels, rq1_y_pred, rq1_average_precision, smell, config, rq1_out_folder, dim, "rnn")
    plot_util.save_precision_recall_curve(rq2_data.eval_labels, rq2_y_pred, rq2_average_precision, smell, config, rq2_out_folder, dim, "rnn")
    tf.keras.backend.clear_session()
    return rq1_auc, rq1_accuracy, rq1_precision, rq1_recall, rq1_f1, rq1_average_precision, \
           rq2_auc, rq2_accuracy, rq2_precision, rq2_recall, rq2_f1, rq2_average_precision, stopped_epoch


def start_training(data, config, conn, smell):
    try:
        return embedding_lstm(data, config, conn, smell)
    except Exception as ex:
        print(ex)
        return ([-1, -1, -1])


def get_all_data(data_path, rq2_eval_data_path, smell):
    print("reading data...")

    train_data, train_labels, eval_data, eval_labels, max_input_length = \
        inputs.get_data(data_path, RQ1_OUT_FOLDER, "rq1_rnn_" + smell,
                                                train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples= 5000)

    train_data = train_data.reshape((len(train_labels), max_input_length))
    eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    rq1_data = input_data.Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)

    training_data, training_labels, eval_data, eval_labels, max_input_length = \
        inputs.get_data_rq2(data_path, rq2_eval_data_path, RQ2_OUT_FOLDER, "rq2_rnn_" + smell,
                        train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=5000)

    training_data = training_data.reshape((len(training_labels), max_input_length))
    eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    rq2_data = input_data.Input_data(training_data, training_labels, eval_data, eval_labels, max_input_length)
    print("reading data... done.")

    return rq1_data, rq2_data


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file(smell, out_folder, model):
    now = datetime.datetime.now()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    return os.path.join(out_folder, model + smell + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main(data_path, rq2_eval_path, smell):
    rq1_data, rq2_data = get_all_data(data_path, rq2_eval_path, smell)

    rnn_layers = {1, 2, 3}
    emb_outputs = [16, 32]
    lstms_units = [32, 64, 128]
    epochs = [2]
    dropouts = [0.2]

    total_iterations = len(emb_outputs) * len(lstms_units) * len(rnn_layers) * len(epochs) * len(dropouts)
    cur_iter = 1
    rq1_outfile = get_out_file(smell, RQ1_OUT_FOLDER, "rnn_rq1_")
    rq2_outfile = get_out_file(smell, RQ2_OUT_FOLDER, "rnn_rq2_")
    write_result(rq1_outfile, "embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    write_result(rq2_outfile, "embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")
    for layer in rnn_layers:
        for emb_output in emb_outputs:
            for lstm_units in lstms_units:
                for epoch in epochs:
                    for dropout in dropouts:
                        print("** Iteration {0} of {1} **".format(cur_iter, total_iterations))
                        config = configuration.RNN_emb_lstm_config(
                            emb_output=emb_output,
                            lstm_units=lstm_units,
                            layers=layer,
                            epochs=epoch,
                            dropout=dropout)
                        try:
                            start_time = time.time()
                            rq1_auc, rq1_accuracy, rq1_precision, rq1_recall, rq1_f1, rq1_average_precision, \
                            rq2_auc, rq2_accuracy, rq2_precision, rq2_recall, rq2_f1, rq2_average_precision, stopped_epoch \
                                = embedding_lstm(rq1_data, rq2_data, config, smell)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            rq1_result_str = (str(emb_output) + ","
                                          + str(layer) + ","
                                          + str(lstm_units) + ","
                                          + str(epoch) + "," + str(stopped_epoch) + "," +
                                          str(rq1_auc) + "," + str(rq1_accuracy) + "," + str(rq1_precision) + "," + str(rq1_recall)
                                         + "," + str(rq1_f1) + "," + str(rq1_average_precision) + "," +
                                         str(elapsed_time) + "\n")
                            rq2_result_str = (str(emb_output) + ","
                                              + str(layer) + ","
                                              + str(lstm_units) + ","
                                              + str(epoch) + "," + str(stopped_epoch) + "," +
                                              str(rq2_auc) + "," + str(rq2_accuracy) + "," + str(
                                        rq2_precision) + "," + str(rq2_recall)
                                              + "," + str(rq2_f1) + "," + str(rq2_average_precision) + "," +
                                              str(elapsed_time) + "\n")
                            write_result(rq1_outfile, rq1_result_str)
                            write_result(rq2_outfile, rq2_result_str)
                            gc.collect()
                        except Exception as ex:
                            print("Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}"
                                  .format(layer, emb_output, lstm_units, epoch))
                            print(ex)
                        cur_iter += 1


if __name__ == "__main__":
    smell_list = {"ComplexMethod", "EmptyCatchBlock", "MagicNumber", "MultifacetedAbstraction"}
    # smell_list = {"ComplexMethod"}

    for smell in smell_list:
        # data_path is for both training and rq1_evaluation
        data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
        rq2_eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
        inputs.preprocess_data(data_path)
        inputs.preprocess_data(rq2_eval_data_path)
        main(data_path, rq2_eval_data_path, smell)
