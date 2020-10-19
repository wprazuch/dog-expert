import os
import pickle

import numpy as np
import tensorflow as tf
from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

from model_utils import get_top_5_predicted_breeds

UPLOAD_FOLDER = r'.\uploads'
MODEL_PATH = r'..\..\model\test_model2'
LOOKUP_PATH = r'..\..\model\class_lookup.pickle'


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = tf.cast(image, tf.uint8)
    return image

def preprocess_image(image):
    image = (image - np.mean(image))/ np.std(image)
    image = image[np.newaxis, ...]
    return image



def create_app(test_config=None):
    # create and configure the app

    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'dogs')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'dogs'))



    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'app.sqlite')
    )


    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MODEL_PATH'] = MODEL_PATH
    app.config['LOOKUP_PATH'] = LOOKUP_PATH


    model = load_model(app.config['MODEL_PATH'])

    with open(app.config['LOOKUP_PATH'], 'rb') as handle:
        class_lookup = pickle.load(handle)
    
    # a simple page that says hello
    @app.route('/predict', methods=['GET', 'POST'])
    def hello():

        if request.method == 'POST':
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            image = load_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = preprocess_image(image)

            result = model.predict(image)[0]
            print(result)
            result_str = str(get_top_5_predicted_breeds(result, class_lookup))


            return render_template("predict.html", prediction_text=result_str)

        return render_template("predict.html")

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                filename)
    
    return app




if __name__ == '__main__':

    

    create_app()
