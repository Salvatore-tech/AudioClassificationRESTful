import json
import os

import keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATASET_ROOT = "/home/saso/Documents/whale_data"
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")
FEATURE_NAME = "mfcc"
TARGET_NAME = "class_label"
BATCH_SIZE = 32

def show_accuracy_loss_plots(history):
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
    plt.show()


# input layer
# keras.layers.Flatten(input_shape=(numeric_features.shape[1])),
def get_model():

    # build network topology
    model = keras.Sequential([

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(2, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def normalize_features(X_train, X_test):
    # fit scaler on training data
    norm = MinMaxScaler().fit(X_train)

    # transform training data
    X_train_norm = norm.transform(X_train)

    # transform testing dataabs
    X_test_norm = norm.transform(X_test)

    return X_train_norm, X_test_norm

if __name__ == "__main__":
    minidf_with_splitted_mfcc_and_zcr = pd.read_csv("./minidf_with_splitted_mfcc_and_zcr.csv")
    X = np.array(minidf_with_splitted_mfcc_and_zcr.iloc[:, 2:])
    y = np.array(minidf_with_splitted_mfcc_and_zcr[[TARGET_NAME]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test = normalize_features(X_train, X_test)

    # build network topology
    model = get_model()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=50)

    model.summary()

    # view history accuracy
    show_accuracy_loss_plots(history)
