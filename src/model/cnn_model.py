import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from cnn_feature_utils import get_features, save_model_to_disk, show_accuracy_loss_plots, \
    show_precision_recall_plots, show_precision_vs_recall, OUTPUT_PATH, get_cnn_with_4_layers

if __name__ == "__main__":
    X, Y = get_features(limit=True)  # limit the dataset load for testing purpouses

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123,
                                                        stratify=Y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    # print(X_train.shape, X_test.shape)

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
    pd.DataFrame.from_dict(history.history).to_csv(os.path.join(OUTPUT_PATH, 'history.csv'), index=False)

    show_accuracy_loss_plots(history, 'accuracy_loss_metrics_test')
    show_precision_recall_plots(history, 'precision_recall_epochs_metrics_test')
    show_precision_vs_recall(history, 'precision_recall_metrics_test')

    y_pred = np.round(model.predict(X_test), 0)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
