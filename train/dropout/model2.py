import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


class CustomModel(object):

    def __init__(self, input_shape=(48, 48, 1), num_classes=7, alpha=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.alpha = alpha

    def make_model(self):
        inputs = layers.Input(shape=self.input_shape, name="Input")

        # Conv2D_1
        x = layers.Conv2D(int(64 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same",
                          kernel_regularizer=l2(0.01))(
            inputs)

        # Conv2D_2
        x = layers.Conv2D(int(64 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # Conv2D_3
        x = layers.Conv2D(int(128 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        # Conv2D_4
        x = layers.Conv2D(int(128 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # Conv2D_5
        x = layers.Conv2D(int(256 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        # Conv2D_6
        x = layers.Conv2D(int(256 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # Conv2D_7
        x = layers.Conv2D(int(512 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        # Conv2D_8
        x = layers.Conv2D(int(512 * self.alpha), kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")(
            x)
        x = layers.BatchNormalization(fused=True)(x)

        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(512, activation=tf.nn.relu)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation=tf.nn.relu)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(128, activation=tf.nn.relu)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=inputs, outputs=x)
