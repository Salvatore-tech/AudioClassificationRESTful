from flask import Flask

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'aiff', 'wav'}
SAMPLE_RATE = 2000

app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"