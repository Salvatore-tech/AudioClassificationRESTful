#!/usr/bin/env python
# coding: utf-8
import os
import sys

import pandas as pd

from feature_builder import FeatureBuilder

DATASET_ROOT = sys.argv[1]
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(os.path.join(DATASET_ROOT, "train.csv"), nrows=10)

    for index, row in df.iterrows():
        full_track_name = TRAINING_PATH + row["clip_name"]
        fb = FeatureBuilder(full_track_name).extract_mfcc().extract_zcr()

    df1 = pd.DataFrame(list(zip(df['clip_name'], df['label'], fb.get_mfcc(), fb.get_zcr())),
                       columns=['track_name', 'target_label', 'mfcc', 'zcr'])

    df_with_splitted_mfccs = pd.DataFrame([pd.Series(x) for x in df1.mfcc])
    df_with_splitted_mfccs.columns = ['mfcc_{}'.format(x + 1) for x in df_with_splitted_mfccs.columns]

    df_with_splitted_mfccs.insert(loc=0, column='track_name', value=df1['track_name'])
    df_with_splitted_mfccs.insert(loc=1, column='class_label', value=df1['target_label'])
    df_with_splitted_mfccs.insert(loc=2, column='zcr', value=df1['zcr'])

    df_with_splitted_mfccs.to_csv('./minidf_with_splitted_mfcc_and_zcr.csv', encoding='utf-8', index=False)
