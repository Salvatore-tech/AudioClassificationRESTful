import os

import numpy as np
from flask import Blueprint, request, flash, url_for, render_template
from werkzeug.utils import secure_filename, redirect
from flask_wtf import FlaskForm
from wtforms import FileField
import librosa as lr
import tensorflow as tf

import shared
from src.shared import UPLOAD_FOLDER, SAMPLE_RATE

ALLOWED_AUDIO_EXTENSIONS = {'aiff'}
classify = Blueprint("classify", __name__)

@classify.route("/upload", methods=['POST'])
def upload_file():
    print('File upload method')

    # check if the post request has the file part
    if 'file' not in request.files:
        #flash('No file part')
        return redirect("/home")

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        # flash('No selected file')
        return redirect("/home")

    if not allowed_file(uploaded_file.filename):
        return redirect("/home")

    filename = secure_filename(uploaded_file.filename)
    fullpath_to_uploaded_file = os.path.join(UPLOAD_FOLDER, filename)
    global current_filename
    current_filename = fullpath_to_uploaded_file
    print('Uploaded new file: ' + filename)
    if not os.path.exists(fullpath_to_uploaded_file):
        uploaded_file.save(fullpath_to_uploaded_file)
    return redirect("/home")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


@classify.route("/predict", methods=['POST'])
def predict_whale():
    audio, _ = lr.load(current_filename, sr=SAMPLE_RATE, res_type='kaiser_fast')
    mel_spectrogram = np.array(get_melspectrogram(audio))
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
    prediction = shared.model.predict_classes(mel_spectrogram).ravel()[0]
    if (np.bool(prediction)):
        flash('Congrats! Whale detected', 'success')
    else:
        flash('That was not a whale, try again', 'failure')
    return redirect("/home")

def get_melspectrogram(audio):
    X = []
    X.append(lr.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=50))
    return X

class MyForm(FlaskForm):
    all = FileField('all')
