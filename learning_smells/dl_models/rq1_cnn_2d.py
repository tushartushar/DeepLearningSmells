from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import tensorflow as tf
import configuration, input_data
import inputs
import plot_util
import time
import metrics_util

# -- Parameters ---
DIM = "2d"

TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
OUT_FOLDER = "../results/rq1/raw"
# TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_cs"
# OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"

TRAIN_VALIDATE_RATIO = 0.7
CLASSIFIER_THRESHOLD = 0.7
# ------------


def cnn(data, config, smell, out_folder=OUT_FOLDER, dim=DIM):
    assert (config.layers >= 1 and config.layers <= 3)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(config.filters, config.kernel, activation='relu',
                                     input_shape=(data.max_input_height, data.max_input_width, 1),
                                     bias_initializer='zeros',
                                     kernel_initializer='random_uniform'
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(config.pooling_window, strides=2))
    for i in range(2, config.layers + 1):
        model.add(tf.keras.layers.Conv2D(2 * config.filters, config.kernel, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(config.pooling_window, strides=2))
        # model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    best_model_filepath = 'weights_best.cnn2d.hdf5'
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
    model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=config.epochs, batch_size=batch_sizes[b_size],
              callbacks=callbacks_list, verbose=1, shuffle=False)

    stopped_epoch = earlystop.stopped_epoch
    model.load_weights(best_model_filepath)

    # y_pred = model.predict_classes(data.eval_data)
    # We manually apply classification threshold
    prob = model.predict_proba(data.eval_data)
    y_pred = inputs.get_predicted_y(prob, CLASSIFIER_THRESHOLD)

    auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = \
        metrics_util.get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)

    # plot_util.save_roc_curve(fpr, tpr, auc, smell, config, out_folder, dim)
    plot_util.save_precision_recall_curve(data.eval_labels, y_pred, average_precision, smell, config, out_folder, dim, "cnn")
    tf.keras.backend.clear_session()
    return auc, accuracy, precision, recall, f1, average_precision, stopped_epoch


def get_all_data(data_path, smell):
    print("reading data...")

    # Load training and eval data
    train_data, train_labels, eval_data, eval_labels, max_input_height, max_input_width = \
        inputs.get_data_2d(data_path, OUT_FOLDER, "rq1_cnn2d_" + smell,
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
    return os.path.join(OUT_FOLDER, "cnn2d_rq1_" + smell +
                        "_" + str(now.strftime("%d%m%Y_%H%M") + ".csv"))


def main(smell, data_path):
    input_data = get_all_data(data_path, smell)

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
                        if cur_iter < 37:
                            cur_iter += 1
                            continue
                        try:
                            config = configuration.CNN_config(layer, filter, kernel, pooling_window, epoch)
                            start_time = time.time()
                            auc, accuracy, precision, recall, f1, average_precision, stopped_epoch = cnn(input_data, config, smell)
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
                        cur_iter += 1


if __name__ == "__main__":
    # smell_list = {"ComplexMethod", "EmptyCatchBlock", "MagicNumber", "MultifacetedAbstraction"}
    smell_list = {"MultifacetedAbstraction"}
    for smell in smell_list:
        data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
        inputs.preprocess_data_2d(data_path)
        main(smell, data_path)
