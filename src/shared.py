import os

from flask import Flask

UPLOAD_FOLDER = './upload'
RESOURCE_FOLDER = './resources'
DL_MODEL_FOLDER = os.path.join(RESOURCE_FOLDER, 'dl_model_saved')
ALLOWED_EXTENSIONS = {'aiff', 'wav'}
SAMPLE_RATE = 2000

app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1000 * 1000 # 5MB is the maximum upload file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

def init():
    global model
    model = {}