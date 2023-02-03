#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import pandas as pd
import librosa as lr
import os
import sys

DATASET_ROOT = sys.argv[1]
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")


class FeatureItems:
    def __init__(self, track_name, class_label, mfcc, zcr):
        self.track_name = track_name
        self.class_label = class_label
        self.mfcc = mfcc
        self.zcr = zcr

    def to_dict(self):
        return {
            'track_name': self.track_name,
            'class_label': self.class_label,
            'mfcc': self.mfcc,
            'zcr': self.zcr
        }


def extract_mfcc(current_audio_obj):
    mfcc = lr.feature.mfcc(y=current_audio_obj)
    mfccs_processed = np.mean(mfcc.T, axis=0)
    return mfccs_processed


def extract_zcr(audio_obj):
    zcr = lr.feature.zero_crossing_rate(audio_obj)
    zcr_processed = np.mean(zcr.T, axis=0)
    return float(zcr_processed[0])


if __name__ == "__main__":
    # Load dataset
    df_with_mfcc = pd.read_csv(os.path.join(DATASET_ROOT, "train.csv"), nrows=10)

    df1 = pd.DataFrame({
        'track_name': [],
        'class_label': [],
        'mfcc': [],
        'zcr': []
    })

    featuresItems = []
    for index, row in df_with_mfcc.iterrows():
        track_name = row["clip_name"]
        class_label = row["label"]
        full_track_name = TRAINING_PATH + row["clip_name"]

        current_audio_obj, _ = lr.load(os.path.join(TRAINING_PATH, full_track_name), res_type='kaiser_fast')

        mfcc = extract_mfcc(current_audio_obj)
        zcr = extract_zcr(current_audio_obj)

        featuresItems.append(FeatureItems(track_name, math.ceil(class_label), mfcc, zcr))

    df1 = pd.DataFrame.from_records([feature.to_dict() for feature in featuresItems])

    df_with_splitted_mfccs = pd.DataFrame([pd.Series(x) for x in df1.mfcc])
    df_with_splitted_mfccs.columns = ['mfcc_{}'.format(x + 1) for x in df_with_splitted_mfccs.columns]

    df_with_splitted_mfccs.insert(loc=0, column='track_name', value=df1['track_name'])
    df_with_splitted_mfccs.insert(loc=1, column='class_label', value=df1['class_label'])
    df_with_splitted_mfccs.insert(loc=2, column='zcr', value=df1['zcr'])

    df_with_splitted_mfccs.to_csv('./minidf_with_splitted_mfcc_and_zcr.csv', encoding='utf-8', index=False)

