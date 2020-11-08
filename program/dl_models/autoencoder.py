from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import regularizers

import input_data
import inputs
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
import configuration

# -- Parameters --
DIM = "1d"
LABELS = ["Normal", "Smell"]

TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"
# TOKENIZER_OUT_PATH = r"..\..\data\tokenizer_out"
# OUT_FOLDER = r"..\results\rq1\raw"
TRAIN_VALIDATE_RATIO = 0.7


# --


def autoencoder_dense(data, smell, layers=1, encoding_dimension=32, epochs=10, with_bottleneck=True, is_final=False, threshold=400000):
    encoding_dim = encoding_dimension
    input_layer = Input(shape=(data.max_input_length,))
    no_of_layers = layers
    prev_layer = input_layer
    for i in range(no_of_layers):
        encoder = Dense(int(encoding_dim / pow(2, i)), activation="relu",
                        activity_regularizer=regularizers.l1(10e-3))(prev_layer)
        prev_layer = encoder
    # bottleneck
    if with_bottleneck:
        prev_layer = Dense(int(encoding_dim / pow(2, no_of_layers)), activation="relu")(prev_layer)
    for j in range(no_of_layers - 1, -1, -1):
        decoder = Dense(int(encoding_dim / pow(2, j)), activation='relu')(prev_layer)
        prev_layer = decoder
    prev_layer = Dense(data.max_input_length, activation='relu')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()

    batch_sizes = [32, 64, 128]
    # batch_sizes = [32, 64, 128, 256, 512]
    b_size = int(len(data.train_data) / batch_sizes[len(batch_sizes) - 1])
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1

    val_split = 0.2
    if is_final:
        val_split = 0
    history = autoencoder.fit(data.train_data,
                              data.train_data,
                              epochs=epochs,
                              # batch_size=batch_size,
                              batch_size=batch_sizes[b_size],
                              verbose=1,
                              validation_split=val_split,
                              shuffle=True).history

    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()

    predictions = autoencoder.predict(data.eval_data)
    mse = np.mean(np.power(data.eval_data - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': data.eval_labels})
    # print(error_df.describe())
    if is_final:
        return find_metrics(error_df, threshold)
    else:
        return find_optimal(error_df)


def autoencoder_cnn(data, config):
    data.train_data = data.train_data.reshape((len(data.train_data), data.max_input_length, 1))
    data.eval_data = data.eval_data.reshape((len(data.eval_labels), data.max_input_length, 1))
    # print("train_data shape: " + str(data.train_data.shape))

    input_layer = Input(shape=(data.max_input_length, 1))
    prev_layer = input_layer
    for i in range(config.layers):
        encoder = Conv1D(int(config.filters / pow(2, i)), config.kernel,
                         activation="relu", #input_shape=(data.max_input_length, 1),
                         padding='same',
                         kernel_initializer='random_uniform')(prev_layer)
        prev_layer = MaxPooling1D((config.pooling_window), strides=config.pooling_window)(encoder)

    # bottleneck
    # prev_layer = Conv1D(int(config.filters / pow(2, config.layers)), config.kernel, activation="relu",
    #                         kernel_initializer='random_uniform')(prev_layer)
    # prev_layer = MaxPooling1D((config.pooling_window), strides=2)(prev_layer)

    # decoder
    for j in range(config.layers - 1, -1, -1):
        prev_layer = Conv1D(int(config.filters / pow(2, j)), config.kernel,
                            padding='same',
                            activation="relu",
                            kernel_initializer='random_uniform')(prev_layer)
        prev_layer = UpSampling1D((config.pooling_window))(prev_layer)
    prev_layer = Dense(1, activation='relu')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()

    # batch_sizes = [32, 64, 128]
    batch_sizes = [32, 64, 128, 256, 512]
    b_size = int(len(data.train_data) / batch_sizes[len(batch_sizes) - 1])
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1
    history = autoencoder.fit(data.train_data,
                              data.train_data,
                              epochs=config.epochs,
                              # batch_size=batch_size,
                              batch_size=batch_sizes[b_size],
                              verbose=1,
                              validation_split=0.2,
                              shuffle=True).history

    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()

    predictions = autoencoder.predict(data.eval_data)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    data.eval_data = data.eval_data.reshape(data.eval_data.shape[0], data.eval_data.shape[1])
    mse = np.mean(np.power(data.eval_data - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': data.eval_labels})
    # print(error_df.describe())
    return find_optimal(error_df)


def find_metrics(error_df, threshold):
    y_pred = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, y_pred)
    precision, recall, f1 = compute_metrics(conf_matrix)
    return threshold, precision, recall, f1


# The following code figures out the optimal threshold
def find_optimal(error_df):
    optimal_threshold = 1000
    max_f1 = 0
    max_pr = 0
    max_re = 0
    for threshold in range(1000, 400000, 5000):
        print("Threshold: " + str(threshold))
        y_pred = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.True_class, y_pred)
        precision, recall, f1 = compute_metrics(conf_matrix)
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold
            max_pr = precision
            max_re = recall
    return optimal_threshold, max_pr, max_re, max_f1


def autoencoder_lstm(data, smell, layers=1, encoding_dimension=8, no_of_epochs=10, with_bottleneck=True,
                     is_final=False):
    data.train_data = data.train_data.reshape((len(data.train_data), data.max_input_length, 1))
    data.eval_data = data.eval_data.reshape((len(data.eval_labels), data.max_input_length, 1))

    encoding_dim = encoding_dimension
    input_layer = Input(shape=(data.max_input_length, 1))
    # input_layer = BatchNormalization()(input_layer)
    no_of_layers = layers
    prev_layer = input_layer
    for i in range(no_of_layers):
        encoder = LSTM(int(encoding_dim / pow(2, i)),
                       # activation="relu",
                       return_sequences=True,
                       recurrent_dropout=0.1,
                       dropout=0.1)(prev_layer)
        prev_layer = encoder
    # bottleneck
    if with_bottleneck:
        prev_layer = LSTM(int(encoding_dim / pow(2, no_of_layers + 1)),
                          # activation="relu",
                          return_sequences=True,
                          recurrent_dropout=0.1,
                          dropout=0.1)(prev_layer)
    for j in range(no_of_layers - 1, -1, -1):
        decoder = LSTM(int(encoding_dim / pow(2, j)),
                       # activation='relu',
                       return_sequences=True,
                       recurrent_dropout=0.1,
                       dropout=0.1)(prev_layer)
        prev_layer = decoder
    prev_layer = TimeDistributed(Dense(1))(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()

    # batch_sizes = [32, 64, 128, 256, 512]
    batch_sizes = [32, 64]
    b_size = int(len(data.train_data) / 512)
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1
    history = autoencoder.fit(data.train_data,
                              data.train_data,
                              epochs=no_of_epochs,
                              batch_size=batch_sizes[b_size],
                              verbose=1,
                              validation_split=0.2,
                              shuffle=True).history

    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()

    predictions = autoencoder.predict(data.eval_data)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    data.eval_data = data.eval_data.reshape(data.eval_data.shape[0], data.eval_data.shape[1])
    mse = np.mean(np.power(data.eval_data - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': data.eval_labels})
    print(error_df.describe())
    return find_optimal(error_df)


def compute_metrics(conf_matrix):
    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    f1 = (2 * precision * recall) / (precision + recall)
    print("precision: " + str(precision) + ", recall: " + str(recall) + ", f1: " + str(f1))
    return precision, recall, f1


def get_all_data(data_path, smell):
    print("reading data...")
    max_eval_samples = 150000
    if smell in ["MultifacetedAbstraction", "FeatureEnvy"]:
        max_eval_samples = 50000

    train_data, eval_data, eval_labels, max_input_length = \
        inputs.get_data_autoencoder(data_path,
                                    train_validate_ratio=TRAIN_VALIDATE_RATIO,
                                    max_training_samples=5000,
                                    max_eval_samples=max_eval_samples,
                                    )
    print("nan count: " + str(np.count_nonzero(np.isnan(train_data))))
    print("train_data: " + str(len(train_data)))
    print("train_data shape: " + str(train_data.shape))
    print("eval_data: " + str(len(eval_data)))
    print("eval_labels: " + str(len(eval_labels)))
    print("reading data... done.")
    return input_data.Input_data(train_data, None, eval_data, eval_labels, max_input_length)


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


def get_out_file(smell, model):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    return os.path.join(OUT_FOLDER, "ae_rq1_" + smell + "_" + model + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main_lstm(smell, data_path, skip_iter=-1):
    input_data = get_all_data(data_path, smell)
    layers = [1, 2]
    encoding_dim = [8, 16, 32]
    epochs = 10
    outfile = get_out_file(smell, "rnn")
    write_result(outfile,
                 "units,threshold,epoch,bottleneck,layer,precision,recall,f1,time\n")
    cur_iter = 1
    for layer in layers:
        for bottleneck in [True]:
            for encoding in encoding_dim:
                if cur_iter <= skip_iter:
                    cur_iter += 1
                    continue
                cur_iter += 1
                write_result(outfile, "processing layer " + str(layer) + " encoding " + str(encoding))
                start_time = time.time()
                # optimal_threshold, max_pr, max_re, max_f1 = autoencoder(input_data, smell, layers=layer,
                #                                                         with_bottleneck=bottleneck)
                try:
                    optimal_threshold, max_pr, max_re, max_f1 = autoencoder_lstm(input_data, smell, layers=layer,
                                                                                 encoding_dimension=encoding,
                                                                                 no_of_epochs=epochs,
                                                                                 with_bottleneck=bottleneck)
                except ValueError as error:
                    print(error)
                    optimal_threshold = -1
                    max_pr = -1
                    max_re = -1
                    max_f1 = -1
                end_time = time.time()
                time_taken = end_time - start_time
                write_result(outfile,
                             str(encoding) + "," + str(optimal_threshold) + "," + str(epochs) + "," + str(bottleneck)
                             + "," + str(layer) + "," +
                             str(max_pr) + "," + str(max_re) + "," + str(max_f1) + "," + str(time_taken) + "\n")


def main_dense(smell, data_path, max_encoding_dim=1024):
    input_data = get_all_data(data_path, smell)
    layers = [1, 2]
    # batch_sizes = [32, 64, 128, 256, 512]
    max_encoding_dimension = min(max_encoding_dim, input_data.max_input_length)
    encoding_dim = [int(max_encoding_dimension / 4), int(max_encoding_dimension / 2), int(max_encoding_dimension)]
    epochs = 20
    outfile = get_out_file(smell, "dense")
    write_result(outfile,
                 "Encoding_dim,threshold,epoch,bottleneck,layer,precision,recall,f1,time\n")
    for layer in layers:
        for bottleneck in [True]:
            for encoding in encoding_dim:
                # for batch_size in batch_sizes:
                start_time = time.time()
                try:
                    optimal_threshold, max_pr, max_re, max_f1 = autoencoder_dense(input_data, smell, layers=layer,
                                                                                  epochs=epochs,
                                                                                  # batch_size=batch_size,
                                                                                  encoding_dimension=encoding,
                                                                                  with_bottleneck=bottleneck)
                except ValueError as error:
                    print(error)
                    optimal_threshold = -1
                    max_pr = -1
                    max_re = -1
                    max_f1 = -1
                end_time = time.time()
                time_taken = end_time - start_time
                write_result(outfile,
                             str(encoding) + "," + str(optimal_threshold) + "," + str(
                                 epochs) + "," + str(bottleneck) + "," + str(layer) + "," +
                             str(max_pr) + "," + str(max_re) + "," + str(max_f1) + "," + str(time_taken) + "\n")


def main_cnn(smell, data_path, max_encoding_dim=1024):
    input_data = get_all_data(data_path, smell)
    filters = [8, 16, 32, 64]
    kernels = [5, 7, 11]
    pooling_windows = [2, 3, 4, 5]
    layers = [1, 2]
    epochs = 20
    outfile = get_out_file(smell, "cnn")
    write_result(outfile, "conv_layers,filters,kernel,max_pooling_window,epochs,precision,recall,f1,time\n")
    for layer in layers:
        for filter in filters:
            for kernel in kernels:
                for pooling_window in pooling_windows:
                    config = configuration.CNN_config(layer, filter, kernel, pooling_window, epochs)
                    start_time = time.time()
                    try:
                        optimal_threshold, max_pr, max_re, max_f1 = autoencoder_cnn(input_data, config)
                    except ValueError as error:
                        print(error)
                        optimal_threshold = -1
                        max_pr = -1
                        max_re = -1
                        max_f1 = -1
                    end_time = time.time()
                    time_taken = end_time - start_time
                    write_result(outfile,
                                 str(layer) + "," + str(filter) + "," + str(
                                     kernel) + "," + str(pooling_window) + "," + str(epochs) + "," +
                                 str(max_pr) + "," + str(max_re) + "," + str(max_f1) + "," + str(time_taken) + "\n")


def main_dense_with_best_params(smell, input_data, layer, epochs=20, encoding=1024, bottleneck=True, threshold=400000):
    # layers = [1, 2]
    # batch_sizes = [32, 64, 128, 256, 512]
    # max_encoding_dimension = min(max_encoding_dim, input_data.max_input_length)
    # encoding_dim = [int(max_encoding_dimension / 4), int(max_encoding_dimension / 2), int(max_encoding_dimension)]
    # epochs = 20
    outfile = get_out_file(smell, "dense_final")
    write_result(outfile,
                 "Encoding_dim,threshold,epoch,bottleneck,layer,precision,recall,f1,time\n")
    # for layer in layers:
    #     for bottleneck in [True]:
    #         for encoding in encoding_dim:
                # for batch_size in batch_sizes:
    start_time = time.time()
    try:
        optimal_threshold, max_pr, max_re, max_f1 = autoencoder_dense(input_data, smell, layers=layer,
                                                                      epochs=epochs,
                                                                      # batch_size=batch_size,
                                                                      encoding_dimension=encoding,
                                                                      with_bottleneck=bottleneck,
                                                                      threshold=threshold,
                                                                      is_final=True)
    except ValueError as error:
        print(error)
        optimal_threshold = -1
        max_pr = -1
        max_re = -1
        max_f1 = -1
    end_time = time.time()
    time_taken = end_time - start_time
    write_result(outfile,
                 str(encoding) + "," + str(optimal_threshold) + "," + str(
                     epochs) + "," + str(bottleneck) + "," + str(layer) + "," +
                 str(max_pr) + "," + str(max_re) + "," + str(max_f1) + "," + str(time_taken) + "\n")


def run_final():
    smell = "ComplexMethod"
    data_path1 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data1 = get_all_data(data_path1, smell)
    main_dense_with_best_params(smell, input_data=input_data1, layer=1,
                                epochs=20, encoding=32, bottleneck=True, threshold=319000)

    smell = "ComplexConditional"
    data_path2 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data2 = get_all_data(data_path2, smell)
    main_dense_with_best_params(smell, input_data=input_data2, layer=1,
                                epochs=20, encoding=16, bottleneck=True, threshold=328000)

    smell = "FeatureEnvy"
    data_path3 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data3 = get_all_data(data_path3, smell)
    main_dense_with_best_params(smell, input_data=input_data3, layer=2,
                                epochs=20, encoding=16, bottleneck=True, threshold=325000)

    smell = "MultifacetedAbstraction"
    data_path4 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data4 = get_all_data(data_path4, smell)
    main_dense_with_best_params(smell, input_data=input_data4, layer=1,
                                epochs=20, encoding=16, bottleneck=True, threshold=328000)


if __name__ == "__main__":
    smell_list = ["ComplexConditional", "ComplexMethod", "MultifacetedAbstraction", "FeatureEnvy"]
    # smell_list = ["FeatureEnvy"]
    # smell_list = ["ComplexConditional"]
    # smell_list = ["ComplexMethod"]

    # for smell in smell_list:
    #     data_path = os.path.join(TOKENIZER_OUT_PATH, smell, DIM)
    #     # main_lstm(smell, data_path)
    #     main_dense(smell, data_path, max_encoding_dim=1024)
    #     main_cnn(smell, data_path, max_encoding_dim=1024)
    run_final()