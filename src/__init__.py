import os

import librosa as lr
import numpy as np
import tensorflow as tf
from flask import flash
from keras.api import keras
from tensorflow.python.keras.models import model_from_json

from src.classify import classify
from src.shared import UPLOAD_FOLDER, SAMPLE_RATE, ALLOWED_EXTENSIONS, app
from views import views

# load json and create model
json_file = open('cnn_model/model_lighter.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('cnn_model/weights_model_lighter.h5')
print("Loaded model from disk")


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return 'file uploaded successfully'
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''

#
# @app.route('/checkWhale/<track_name>', methods=['GET'])
# def check_whale(track_name):
#     full_path_to_track = os.path.join(UPLOAD_FOLDER, track_name)
#     if not os.path.isfile(full_path_to_track):
#         flash('File does not exists')
#         return 'File does not exists'
#     else:
#         audio, _ = lr.load(full_path_to_track, sr=SAMPLE_RATE, res_type='kaiser_fast')
#         mel_spectrogram = np.array(get_melspectrogram(audio))
#         mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
#         prediction = np.round(model.predict(mel_spectrogram), 0)
#         if prediction == 1:
#             return 'That is a whale'
#         return 'That is not a whale'
#
#
# def allowed_file(filename):
#     return '.' in filename and \
#         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def get_melspectrogram(audio):
#     X = []
#     X.append(lr.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=50))
#     return X


if __name__ == '__main__':
    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(classify, url_prefix="/classify")

    # optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'
                           ])
    app.run(debug=True)
