from keras.api import keras
from tensorflow.python.keras.models import model_from_json

import shared
from src.classify import classify
from src.shared import *
from views import views

MODEL_ARCHITECTURE = 'model_lighter.json'
TRAINED_MODEL_WEIGHTS = 'weights_model_lighter.h5'

if __name__ == '__main__':
    shared.init()

    # load json and create model
    json_file = open(os.path.join(DL_MODEL_FOLDER, MODEL_ARCHITECTURE), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    shared.model = model_from_json(loaded_model_json)

    # load weights into new model
    shared.model.load_weights(os.path.join(DL_MODEL_FOLDER, TRAINED_MODEL_WEIGHTS), 'r')
    print("Loaded model from disk")

    # register blueprints
    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(classify, url_prefix="/classify")

    shared.model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy,
                         metrics=['accuracy'
                                  ])
    app.run(debug=True)
