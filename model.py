import json
import os

import keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml_model import *

DATASET_ROOT = "/home/saso/Documents/whale_data"
TRAINING_PATH = os.path.join(DATASET_ROOT, "train/")
FEATURE_NAME = "mfcc"
TARGET_NAME = "class_label"
BATCH_SIZE = 32

def normalize_features(X_train, X_test):
    # fit scaler on training data
    norm = MinMaxScaler().fit(X_train)

    # transform training data
    X_train_norm = norm.transform(X_train)

    # transform testing dataabs
    X_test_norm = norm.transform(X_test)

    return X_train_norm, X_test_norm

if __name__ == "__main__":
    df_with_splitted_features = pd.read_csv("./support/df_with_splitted_features_fixed.csv")

    models = {}
    models['Logistic Regression'] = LogisticRegression(random_state=42)
    models['SVM'] = LinearSVC(random_state=42)
    models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
    models['Random Forest'] = RandomForestClassifier(random_state=42)
    models['Linear Discriminant Analysis'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()

    X = np.array(df_with_splitted_features.iloc[:, 2:])
    y = np.array(df_with_splitted_features[[TARGET_NAME]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = normalize_features(X_train, X_test)

    accuracy, precision, recall = {}, {}, {}
    for key in models.keys():
        print(key)
        models[key].fit(X_train, y_train.ravel())

        predictions = models[key].predict(X_test)

        accuracy[key] = accuracy_score(y_true=y_test, y_pred=predictions)
        precision[key] = precision_score(y_true=y_test, y_pred=predictions)
        recall[key] = recall_score(y_true=y_test, y_pred=predictions)

    df_metrics = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_metrics['Accuracy'] = accuracy.values()
    df_metrics['Precision'] = precision.values()
    df_metrics['Recall'] = recall.values()
    df_metrics.to_csv('./model_metrics_fullfeature.csv', encoding='utf-8')
