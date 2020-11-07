import dog_expert
from dog_expert import DataLoader, Preprocessor
import tensorflow as tf
from dog_expert import models

from tensorflow.keras.optimizers import Adam
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8


def main():

    with open(r'artifacts/class_mapping.pkl', 'rb') as handle:
        class_mapping = pickle.load(handle)

    NO_CLASSES = len(class_mapping.keys())

    preprocessor = Preprocessor()
    preprocessor.cast(dtype=tf.float32).normalize()

    data_loader = DataLoader(train_dataset_dir=r'datasets\bing',
                             val_dataset_dir=r'datasets\stanford', batch_size=BATCH_SIZE,
                             class_mapping=class_mapping, preprocessor=preprocessor)

    train = data_loader.train_dataset
    val = data_loader.val_dataset

    model = models.Xception(num_classes=NO_CLASSES, image_size=IMAGE_SIZE)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        learning_rate=0.003), metrics=['accuracy'])

    history = model.fit(train, validation_data=val,
                        epochs=50)

    model.save('test_model1.h5')

    with open('history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()
