import dog_expert
from dog_expert import DataLoader, Preprocessor
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception


IMAGE_SIZE = (256, 256)


def main():
    preprocessor = Preprocessor()
    preprocessor.cast(dtype=tf.float32).normalize()

    class_mapping = {'affenpinscher': 0,
                     'afghan_hound': 1,
                     'african_hunting_dog': 2,
                     'airedale': 3,
                     'american_staffordshire_terrier': 4,
                     'appenzeller': 5,
                     'australian_terrier': 6,
                     'basenji': 7,
                     'basset': 8}

    data_loader = DataLoader(dataset_dir=r'datasets\stanford2',
                             batch_size=8, class_mapping=class_mapping, preprocessor=preprocessor)

    train = data_loader.train_dataset
    val = data_loader.val_dataset

    print(len(train))
    print(len(val))

    base_model = Xception(weights=None, include_top=False, input_shape=(*IMAGE_SIZE, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(9, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        learning_rate=0.03), metrics=['accuracy'])

    history = model.fit(train, validation_data=val,
                        epochs=50)
    # callbacks=[checkpoint])


if __name__ == '__main__':

    main()
