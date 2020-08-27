import mlflow
import argparse
import sys
from mlflow import pyfunc
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
import mlflow.tensorflow

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

def main(argv):

    model1 = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(10)
            ])

    model2 = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(10)
            ])


    for epoch, model in zip([10, 15], [model1, model2]):

        with mlflow.start_run():

            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", default=100, type=int, help="batch size")
            parser.add_argument("--train_steps", default=1000, type=int, help="number of training steps")

            args = parser.parse_args(argv[1:])

            fashion_mnist = keras.datasets.fashion_mnist

            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

            train_images = train_images / 255.0

            test_images = test_images / 255.0

            model1 = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(10)
            ])


            mlflow.tensorflow.autolog(every_n_iter=1)

            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", default=100, type=int, help="batch size")
            parser.add_argument("--train_steps", default=1000, type=int, help="number of training steps")



            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=".\logs")

            history = model.fit(train_images, train_labels, epochs=epoch, validation_split=0.2) #, callbacks=[tensorboard_callback])



if __name__ == "__main__":
    main(sys.argv)