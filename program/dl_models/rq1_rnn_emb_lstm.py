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
from sklearn.dummy import DummyClassifier

# --- Parameters --
DIM = "1d"
C2V = True # It means whether we are analyzing plain source code that is tokenized (False) or vectors from Code2Vec (True)

if C2V:
    # TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/c2v_vectors/"
    # OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"
    TOKENIZER_OUT_PATH = r"..\..\data\c2v_vectors"
    OUT_FOLDER = r"..\results\rq1\raw"
else:
    # TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
    # OUT_FOLDER = "../results/rq1/raw_temp"
    TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"

TRAIN_VALIDATE_RATIO = 0.7
CLASSIFIER_THRESHOLD = 0.7


# ---


def embedding_lstm(data, config, smell, out_folder=OUT_FOLDER, dim=DIM, iteration=0, is_final=False):
    tf.keras.backend.clear_session()
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
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
    best_model_filepath = 'weights_best.rnn.' + smell + str(iteration) + '.hdf5'
    if os.path.exists(best_model_filepath):
        print("deleting the old weights file..")
        os.remove(best_model_filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks_list = [earlystop, checkpoint]

    batch_sizes = [32, 64, 128, 256]
    b_size = int(len(data.train_labels) / 512)
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1

    if is_final:
        model.fit(data.train_data,
                  data.train_labels,
                  epochs=config.epochs,
                  batch_size=batch_sizes[b_size])
        stopped_epoch = config.epochs

    else:
        model.fit(data.train_data,
                  data.train_labels,
                  validation_split=0.2,
                  epochs=config.epochs,
                  batch_size=batch_sizes[b_size],
                  callbacks=callbacks_list)
        stopped_epoch = earlystop.stopped_epoch
        model.load_weights(best_model_filepath)

    # y_pred = model.predict(data.eval_data).ravel()
    # y_pred = model.predict_classes(data.eval_data)
    # We manually apply classification threshold
    prob = model.predict_proba(data.eval_data)
    y_pred = inputs.get_predicted_y(prob, CLASSIFIER_THRESHOLD)

    auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = \
        metrics_util.get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)

    if is_final:
        plot_util.save_roc_curve(fpr, tpr, auc, smell, config, out_folder, DIM)
        plot_util.save_precision_recall_curve(data.eval_labels, y_pred, average_precision, smell, config, out_folder,
                                              dim, "rnn")
    tf.keras.backend.clear_session()
    return auc, accuracy, precision, recall, f1, average_precision, stopped_epoch


def start_training(data, config, conn, smell):
    try:
        return embedding_lstm(data, config, conn, smell)
    except Exception as ex:
        print(ex)
        return ([-1, -1, -1])


def get_all_data(data_path, smell):
    print("reading data...")

    if smell in ["ComplexConditional", "ComplexMethod"]:
        max_eval_samples = 150000  # for impl smells (methods)
    else:
        max_eval_samples = 50000  # for design smells (classes)

    train_data, train_labels, eval_data, eval_labels, max_input_length = \
        inputs.get_data(data_path,
                        train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=5000,
                        max_eval_samples=max_eval_samples, is_c2v=C2V)

    train_data = train_data.reshape((len(train_labels), max_input_length))
    eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    print("reading data... done.")
    return input_data.Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file(smell):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    if C2V:
        c2v = 'c2v'
    else:
        c2v = ''
    return os.path.join(OUT_FOLDER, "rnn_rq1_" + smell + "_" + c2v
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main(data_path, smell, skip_iter=-1, iterations_to_process=100):
    data = get_all_data(data_path, smell)

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
                            auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = embedding_lstm(
                                data, config, smell, iteration=cur_iter)
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
                            gc.collect()
                        except Exception as ex:
                            print("Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}"
                                  .format(layer, emb_output, lstm_units, epoch))
                            print(ex)
                            write_result(outfile, str(emb_output) + ","
                                         + str(layer) + ","
                                         + str(lstm_units) + ","
                                         + str(epoch) + ",-1,-1,-1,-1,-1,-1,-1\n")
                        cur_iter += 1
                        if cur_iter > iterations_to_process:
                            print("Done with the specified number of iterations.")
                            return


def run_rnn_with_best_params(smell, input_data, rnn_layers, emb_output, lstm_units, epochs):
    config = configuration.RNN_emb_lstm_config(
        emb_output=emb_output,
        lstm_units=lstm_units,
        layers=rnn_layers,
        epochs=epochs,
        dropout=0.2)
    outfile = get_out_file(smell + "final")
    write_result(outfile,
                 "embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,time\n")

    try:
        start_time = time.time()
        auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = embedding_lstm(input_data, config,
                                                                                                smell, is_final=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        result_str = (str(emb_output) + ","
                      + str(rnn_layers) + ","
                      + str(lstm_units) + ","
                      + str(epochs) + "," + str(stopped_epoch) + "," +
                      str(auc) + "," + str(accuracy) + "," + str(precision) + "," + str(recall)
                      + "," + str(f1) + "," + str(average_precision) + "," +
                      str(elapsed_time) + "\n")
        write_result(outfile, result_str)
        gc.collect()
    except Exception as ex:
        print("Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}"
              .format(rnn_layers, emb_output, lstm_units, epochs))
        print(ex)
        write_result(outfile, str(emb_output) + ","
                     + str(rnn_layers) + ","
                     + str(lstm_units) + ","
                     + str(epochs) + ",-1,-1,-1,-1,-1,-1,-1\n")


def run_final():
    smell = "ComplexMethod"
    data_path1 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data1 = get_all_data(data_path1, smell)
    run_rnn_with_best_params(smell, input_data=input_data1, emb_output=32, rnn_layers=1, lstm_units=64, epochs=24)

    smell = "ComplexConditional"
    data_path2 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data2 = get_all_data(data_path2, smell)
    run_rnn_with_best_params(smell, input_data=input_data2, emb_output=32, rnn_layers=1, lstm_units=64, epochs=3)

    smell = "FeatureEnvy"
    data_path3 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data3 = get_all_data(data_path3, smell)
    run_rnn_with_best_params(smell, input_data=input_data3, emb_output=16, rnn_layers=2, lstm_units=64, epochs=16)

    smell = "MultifacetedAbstraction"
    data_path4 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data4 = get_all_data(data_path4, smell)
    run_rnn_with_best_params(smell, input_data=input_data4, emb_output=16, rnn_layers=2, lstm_units=128, epochs=11)


def measure_random_performance():
    smell_list = {"ComplexMethod", "EmptyCatchBlock", "MagicNumber", "MultifacetedAbstraction"}

    outfile = get_out_file("random_classifier")
    write_result(outfile, "smell,auc,precision,recall,f1,average_precision\n")
    for smell in smell_list:
        data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
        input_data = get_all_data(data_path, smell)
        y_pred = np.random.randint(2, size=len(input_data.eval_labels))

        auc, precision, recall, f1, average_precision, fpr, tpr = \
            metrics_util.get_all_metrics_(input_data.eval_labels, y_pred)

        write_result(outfile,
                     smell + "," + str(auc) + "," + str(precision) + "," + str(recall) + "," + str(f1) + "," + str(
                         average_precision) + "\n")


def measure_performance_dummy_classifier():
    outfile = get_out_file("dummy_classifier")
    write_result(outfile, "smell,auc,precision,recall,f1,average_precision\n")
    for smell in smell_list:
        data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
        input_data = get_all_data(data_path, smell)
        # clf = DummyClassifier(strategy='stratified', random_state=0)
        clf = DummyClassifier(strategy='most_frequent', random_state=0)
        inverted_train_labels = inputs.invert_labels(input_data.train_labels)

        # clf.fit(input_data.train_data, input_data.train_labels)
        clf.fit(input_data.train_data, inverted_train_labels)
        y_pred = clf.predict(input_data.eval_data)

        auc, precision, recall, f1, average_precision, fpr, tpr = \
            metrics_util.get_all_metrics_(input_data.eval_labels, y_pred)

        write_result(outfile,
                     smell + "," + str(auc) + "," + str(precision) + "," + str(recall) + "," + str(f1) + "," + str(
                         average_precision) + "\n")


if __name__ == "__main__":
    # smell_list = ["ComplexConditional", "ComplexMethod", "MultifacetedAbstraction", "FeatureEnvy"]

    # smell_list = {"ComplexMethod"}

    # for smell in smell_list:
    #     data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    #     inputs.preprocess_data(data_path)
    # main(data_path, smell)

    # The following is the last step to get the final results. hyper parameters seletected from the best performance (f1)
    run_final()

    # Generate baseline using random classifier
    # measure_random_performance()

    # Let's say what a dummy classifier says
    # measure_performance_dummy_classifier()
