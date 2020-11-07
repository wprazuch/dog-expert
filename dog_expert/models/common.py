
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential


def Xception(num_classes: int, image_size=(256, 256)):

    base_model = applications.Xception(
        weights=None, include_top=False, input_shape=(*image_size, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes)(x)
    predictions = layers.Activation('softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
