import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Concatenate, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
import numpy as np

class ArNetModel:

    def __init__(self, input_shape, output_size, dropout_rate=0.2, padding='same', activation=tf.nn.relu):

        self._input_shape = input_shape
        self._output_size = output_size
        self._dropout_rate = dropout_rate
        self._padding = padding
        self._activation = activation

        self._build()

    def _build(self):

        inputs = Input(self._input_shape)
        x = Conv2D(32, (3, 3), padding=self._padding, activation=self._activation)(inputs)
        x = Conv2D(32, (3, 3), padding=self._padding, activation=self._activation)(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding=self._padding, activation=self._activation)(x)
        x = Conv2D(64, (3, 3), padding=self._padding, activation=self._activation)(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(128, (3, 3), padding=self._padding, activation=self._activation)(x)
        x = Conv2D(128, (3, 3), padding=self._padding, activation=self._activation)(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)

        skip = Flatten()(inputs)
        skip = Dense(512, activation=self._activation)(skip)
        skip = Dense(256, activation=self._activation)(skip)

        concat = x

        concat = Flatten()(concat)
        concat = Concatenate()([concat, skip])
        concat = Dense(512, activation=self._activation)(concat)
        concat = BatchNormalization()(concat)
        concat = Dropout(0.5)(concat)
        concat = Dense(256, activation=self._activation)(concat)
        concat = BatchNormalization()(concat)
        concat = Dropout(0.5)(concat)
        concat = Dense(128, activation=self._activation)(concat)
        concat = BatchNormalization()(concat)
        concat = Dropout(0.5)(concat)
        concat = Dense(64, activation=self._activation)(concat)
        concat = BatchNormalization()(concat)
        concat = Dropout(0.5)(concat)
        concat = Dense(self._output_size, activation=tf.nn.softmax)(concat)
        model = Model(inputs=inputs, outputs=concat)

        self.model = model

    def plot_model(self):
        plot_model(self.model, 'model_architecture.png', show_shapes=True, show_layer_names=False, dpi=150, show_layer_activations=True)

    def summary(self):
        self.model.summary()
