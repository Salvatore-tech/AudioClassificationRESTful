import os
import sys

import librosa as lr
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.api import keras
from keras.callbacks import EarlyStopping
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

JPG_EXTENSION = '.jpg'

SAMPLE_RATE = 2000

DATASET_ROOT = sys.argv[1]
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")
OUTPUT_PATH = './output'


def show_accuracy_loss_plots(history, filename):
    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    plt.savefig(os.path.join(OUTPUT_PATH, filename + JPG_EXTENSION))


def show_precision_recall_plots(history, filename):
    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["precision_m"], label="train precision")
    axs[0].plot(history.history["val_accuracy"], label="test precision")
    axs[0].set_ylabel("Precision")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Precision eval")

    # create error sublpot
    axs[1].plot(history.history["recall_m"], label="train recall")
    axs[1].plot(history.history["val_recall_m"], label="test recall")
    axs[1].set_ylabel("Recall")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Recall eval")
    plt.savefig(os.path.join(OUTPUT_PATH, filename + JPG_EXTENSION))


def show_precision_vs_recall(history, filename):
    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["precision_m"], label="train precision")
    axs[0].plot(history.history["recall_m"], label="train recall")
    axs[0].set_ylabel("Precision")
    axs[0].set_xlabel("Recall")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Precision/recall train")

    # create error sublpot
    axs[1].plot(history.history["val_precision_m"], label="test precision")
    axs[1].plot(history.history["val_recall_m"], label="test recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_xlabel("Recall")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Precision/Recall test")
    plt.savefig(os.path.join(OUTPUT_PATH, filename + JPG_EXTENSION))


def get_melspectrogram(audio):
    return lr.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=50)


def visualize_melSpectrum(melSpectrum, sr):
    melSpectrum_db = librosa.power_to_db(melSpectrum, ref=np.max)
    librosa.display.specshow(melSpectrum_db, y_axis='mel', x_axis='time', sr=sr, cmap='magma');
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_simple_cnn():
    input_shape = (50, 8, 1)
    CNNmodel = models.Sequential()
    CNNmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNNmodel.add(MaxPooling2D((2, 2)))

    CNNmodel.add(Flatten())
    CNNmodel.add(Dense(32, activation='relu'))
    CNNmodel.add(Dense(1, activation='sigmoid'))
    CNNmodel.summary()
    CNNmodel.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['acc', f1_m, precision_m, recall_m])
    return CNNmodel

def get_cnn_with_4_layers():
    input_shape = (50, 8, 1)
    CNNmodel = models.Sequential()

    CNNmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
    CNNmodel.add(MaxPooling2D((2, 2), padding='same'))
    CNNmodel.add(Dropout(0.25))

    # The second convolution
    CNNmodel.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    CNNmodel.add(MaxPooling2D((2, 2), padding='same'))
    CNNmodel.add(Dropout(0.25))

    # The third convolution
    CNNmodel.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    CNNmodel.add(MaxPooling2D((2, 2), padding='same'))
    CNNmodel.add(Dropout(0.25))

    # Flatten the results to feed into a DNN
    CNNmodel.add(Flatten())

    # 512 neuron hidden layer
    CNNmodel.add(Dense(512, activation='relu'))
    CNNmodel.add(Dropout(0.5))

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
    CNNmodel.add(Dense(1, activation='sigmoid'))

    print(CNNmodel.summary())

    optimizer = keras.optimizers.Adam(lr=0.001)
    CNNmodel.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,
                     metrics=['accuracy',
                              f1_m, precision_m, recall_m
                              ])
    return CNNmodel


def save_model_to_disk(model):
    model.save("ligher_model")
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open(os.path.join(OUTPUT_PATH, "model_lighter.json"), "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights(os.path.join(OUTPUT_PATH, "weights_model_lighter.h5"))
    print("Saved model to disk")


def get_features(limit=False, unsupervisioned=False):
    global X, Y
    X = []
    Y = []
    # Load dataset
    if limit:
        df = pd.read_csv(os.path.join(DATASET_ROOT, "train.csv"), nrows=1000)
    else:
        df = pd.read_csv(os.path.join(DATASET_ROOT, "train.csv"))
    if(unsupervisioned):
        for index, row in df.iterrows():
            full_track_name = TRAINING_PATH + row["clip_name"]
            audio, _ = lr.load(full_track_name, sr=SAMPLE_RATE, res_type='kaiser_fast')
            X.append(np.mean(get_melspectrogram(audio), axis=1))
        return X
    else:
        for index, row in df.iterrows():
            full_track_name = TRAINING_PATH + row["clip_name"]
            audio, _ = lr.load(full_track_name, sr=SAMPLE_RATE, res_type='kaiser_fast')
            X.append(get_melspectrogram(audio))
            Y.append(row["label"])
        return X, Y


if __name__ == "__main__":
    get_features(True)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123,
                                                        stratify=Y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    print(X_train.shape, X_test.shape)

    model = get_cnn_with_4_layers()

    # early stopping callback
    # This callback will stop the training when there is no improvement in
    # the validation loss for 10 consecutive epochs.
    es = EarlyStopping(monitor='recall_m',
                       mode='max',  # don't minimize the accuracy!
                       patience=10,
                       restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        callbacks=es,
                        epochs=100, validation_data=(X_test, y_test), shuffle=True,
                        verbose=1)

    save_model_to_disk(model)
    pd.DataFrame.from_dict(history.history).to_csv(os.path.join(OUTPUT_PATH, 'history_lighter.csv'), index=False)

    show_accuracy_loss_plots(history)
    show_precision_recall_plots(history)
    show_precision_vs_recall(history)

    y_pred = np.round(model.predict(X_test), 0)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
