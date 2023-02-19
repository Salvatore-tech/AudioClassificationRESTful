import os

from flask import Blueprint, request, flash, url_for, render_template
from werkzeug.utils import secure_filename, redirect
from flask_wtf import FlaskForm
from wtforms import FileField

from src.shared import UPLOAD_FOLDER

ALLOWED_AUDIO_EXTENSIONS = {'aiff'}
classify = Blueprint("classify", __name__)
current_filename = ''

@classify.route("/upload", methods=['POST'])
def upload_file():
    print('File upload method')

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect("/home")

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        flash('No selected file')
        return redirect("/home")

    if not allowed_file(uploaded_file.filename):
        return redirect("/home")

    filename = secure_filename(uploaded_file.filename)
    fullpath_to_uploaded_file = os.path.join(UPLOAD_FOLDER, filename)
    current_filename = fullpath_to_uploaded_file
    print('Uploaded new file: ' + filename)
    if not os.path.exists(fullpath_to_uploaded_file):
        uploaded_file.save(fullpath_to_uploaded_file)
    return redirect("/home")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


@classify.route("/detect")
def submit():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return 'file uploaded successfully'


class MyForm(FlaskForm):
    all = FileField('all')
