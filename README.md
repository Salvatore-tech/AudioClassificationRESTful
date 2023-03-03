# AudioClassificationRESTful
This repository contains the implementation of a Deep Learning Network used to to detect North Atlantic right whale calls from audio recordings, prevent collisions with shipping traffic.
We depend on shipping industry's uninterrupted ability to transport goods across long distances. Navigation technologies combine accurate position and environmental data to calculate optimal transport routes.
Reducing the impact of commercial shipping on the ocean’s environment, while achieving commercial sustainability, is of increasing importance, especially as it relates to the influence of cumulative noise “footprints” on the great whales.

## Data description
The data consists of 30,000 training samples splitted using 80/20 rule for training and validation sets.
Each candidate is a 2-second .aiff sound clip with a sample rate of 2 kHz. The file "train.csv" gives the labels for the train set.
Candidates that contain a right whale call have label=1, otherwise label=0.

## Implementation details
The problem is solved by building and training a 4-layers Convolutional Neural Network trained for 100 epochs with an Early Stopping on recall metric. 
Mel Frequency Cepstral Coefficients have been extracted as feature of interest to perform the classification, achieving a 91% accuracy and 86% recall on validation data.

A minimalistic interface is provided using the Flask microframework to build Rest endpoints and bind the backend to a basic template.
In this way, the user is able to upload a file, from "sample" folder, and detect the whale calls.

![cnn_4_accuracy](https://user-images.githubusercontent.com/59176654/222837745-970779d0-6a93-4380-93ba-9b0cc7df8b4a.jpg)


## Reference 
https://www.kaggle.com/competitions/whale-detection-challenge/overview
