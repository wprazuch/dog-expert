import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pathlib

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import save_img


def load_by_classes(input_path, no_images=None):
    data = []
    labels = []
    for class_dir in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, class_dir)
        if no_images is None:
            files = os.listdir(path)
        else:
            files = os.listdir(path)[:no_images]
        for file in files:
            img = load_img(pathlib.Path(path, file))
            img = np.array(img)
            data.append(img)
            labels.append(class_dir)
    return data, labels


def resize_images_from(input_path, image_size):
    for class_dir in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, class_dir)
        files = os.listdir(path)
        for file in files:
            img = load_img(pathlib.Path(path, file))
            img = np.array(img)
            img = resize(img, image_size).numpy()
            save_img(pathlib.Path(path, file), img)


def preprocess_input(data, labels):
    for i in tqdm(range(len(data))):
        try:
            data[i] = data[i].astype(np.float32)
            data[i] = resize(data[i], IMAGE_SIZE).numpy()
            #data[i] = rgb2gray(data[i])
            data[i] = 1./255. * data[i]
        except:
            del data[i]
            del labels[i]
    return data, labels


def load_data(path, labels_df=None):
    data = []
    labels = []
    for file in tqdm(os.listdir(path)):
        img = load_img(pathlib.Path(path, file))
        img = np.array(img)
        data.append(img)
        if labels_df is not None:
            target = labels_df.loc[labels_df['id'].str.startswith(
                file.split('.')[0]), 'target'].values[0]
            labels.append(target)
    return data, labels
