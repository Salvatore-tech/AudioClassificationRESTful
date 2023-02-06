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
        fb = FeatureBuilder(full_track_name).extract_mfcc().extract_zcr().extract_chroma().extract_rms().extract_mel()

    df1 = pd.DataFrame(list(zip(df['clip_name'], df['label'], fb.get_mfcc(), fb.get_zcr(), fb.get_chroma(), fb.get_rms(), fb.get_mel())),
                       columns=['track_name', 'target_label', 'mfcc', 'zcr', 'chroma', 'rms', 'mel'])

    df_with_splitted_features = pd.DataFrame()

    df_splitted_mfcc = pd.DataFrame([pd.Series(x) for x in df1.mfcc])
    df_splitted_mfcc.columns = ['mfcc_{}'.format(x + 1) for x in df_splitted_mfcc.columns]

    df_splitted_chroma = pd.DataFrame([pd.Series(x) for x in df1.chroma])
    df_splitted_chroma.columns = ['chroma_{}'.format(x + 1) for x in df_splitted_chroma.columns]

    # TODO: skipping mels for the moment
    # df_splitted_mel = pd.DataFrame([pd.Series(x) for x in df1.mel])
    # df_splitted_mel.columns = ['mel_{}'.format(x + 1) for x in df_splitted_mel.columns]

    df_with_splitted_features.insert(loc=0, column='track_name', value=df1['track_name'])
    df_with_splitted_features.insert(loc=1, column='class_label', value=df1['target_label'])
    df_with_splitted_features.insert(loc=2, column='zcr', value=df1['zcr'])
    df_with_splitted_features.insert(loc=3, column='rms', value=df1['rms'])
    df_with_splitted_features = pd.concat([df_with_splitted_features, df_splitted_mfcc, df_splitted_chroma], axis=1)

    df_with_splitted_features.to_csv('./support/minidf_with_splitted_features.csv', encoding='utf-8', index=False)
