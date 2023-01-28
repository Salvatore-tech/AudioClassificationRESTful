#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import librosa as lr
import os
import json
from sklearn.model_selection import train_test_split
import keras as keras

DATASET_ROOT = "/home/saso/Documents/whale_data"
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")
SAMPLE_RATE = 22050
N_FFT = 2048

# dictionary to store mapping, labels, and MFCCs
data = {
    "track_name": [],
    "target_label": [],
    "mfcc": []
}


def extract_mfcc(file_name, num_mfcc=13, n_fft=2048, hop_length=512):
    audio, sample_rate = lr.load(os.path.join(TRAINING_PATH, file_name), res_type='kaiser_fast')
    mfcc = lr.feature.mfcc(audio, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc


def save_to_dict(dict, key, value):
    dict[key].append(value)


def export_to_json(dict, json_path):
    with open(json_path, "w") as fp:
        json.dump(dict, fp, indent=4)


def load_dict_from_json(data_path, feature_name, target_name):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data[feature_name])
    y = np.array(data[target_name])

    print("Data succesfully loaded!")

    return X, y


if __name__ == "__main__":
    # Load dataset
    train_df = pd.read_csv(os.path.join(DATASET_ROOT, "train.csv"), nrows=10)

    for index, row in train_df.iterrows():
        current_track_name = row["clip_name"]
        current_target_label = row["label"]
        current_mfcc = extract_mfcc(current_track_name)
        save_to_dict(data, "track_name", current_track_name)
        save_to_dict(data, "target_label", current_target_label)
        save_to_dict(data, "mfcc", current_mfcc.tolist())

    # export_to_json(data, "data.json")

    # create train/test split
    X = np.array(data["mfcc"])
    y = np.array(data["target_label"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
