import os
import tensorflow as tf
from .Preprocessor import Preprocessor

import numpy as np
from tensorflow.keras.preprocessing.image import load_img


opj = os.path.join


class DataLoader:

    buffer_size = tf.data.experimental.AUTOTUNE

    def __init__(self, dataset_dir: str, batch_size: int, class_mapping: dict,
                 preprocessor: Preprocessor = None):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.class_mapping = class_mapping

        self.train_dataset = None
        self.val_dataset = None

        self.preprocessor = Preprocessor() if preprocessor is None else preprocessor

    @property
    def train_dataset(self):
        return self.__batch_and_prefetch(self.__train_dataset)

    @train_dataset.setter
    def train_dataset(self, dataset: tf.data.Dataset):
        self.__train_dataset = dataset

    @property
    def val_dataset(self):
        return self.__batch_and_prefetch(self.__val_dataset)

    @val_dataset.setter
    def val_dataset(self, dataset: tf.data.Dataset):
        self.__val_dataset = dataset

    @property
    def preprocessor(self) -> Preprocessor:
        return self.__preprocessor

    @preprocessor.setter
    def preprocessor(self, preprocessor: Preprocessor):
        self.__preprocessor = preprocessor
        self.__reinstantiate()

    def __batch_and_prefetch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.padded_batch(
            self.batch_size,
            padded_shapes=({'data_point': [None, None, 3]},
                           {'target': [None, self.no_classes]})).prefetch(
            buffer_size=self.buffer_size)

    def __reinstantiate(self):
        self.train_dataset = self.__create_dataset_pipeline(opj(self.dataset_dir))
        self.val_dataset = self.__create_dataset_pipeline(opj(self.dataset_dir), shuffle=False)

    def __get_dataset_filepaths(self, path: str) -> list:
        labelled_pairs = []

        dirs = os.listdir(path)
        for dirr in dirs:
            full_p = opj(path, dirr)
            labelled_pairs_for_class = [(opj(full_p, file), str(self.class_mapping[dirr]))
                                        for file in os.listdir(full_p)]

            labelled_pairs.extend(labelled_pairs_for_class)
        return labelled_pairs

    def __create_dataset_pipeline(self, path, shuffle=True) -> tf.data.Dataset:

        def load_image(path):
            img = load_img(path)
            img = np.array(img)
            return img

        def get_one_hot(idx):
            one_hot = np.zeros((len(self.class_mapping.keys())))
            one_hot[idx] = 1
            return one_hot

        def load_image_label_pairs(case):
            return {'data_point': load_image(case[0]),
                    'target': float(get_one_hot(int(case[1])), out_type=tf.float32)}

        dataset = self.__get_dataset_filepaths(path)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(load_image_label_pairs)
        dataset = self.processor.add_to_graph(dataset)

        if shuffle:
            dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)

        return dataset
