import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Input,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Simple function so we can keep tensorflow imports contained to one file
def clone_model(model):
    return tf.keras.models.clone_model(model)


def build_dqn(possible_actions, learning_rate):
    inputs = Input(shape=(84, 84, 4))

    # "float16" provides better performance on Turing GPUs
    x = tf.cast(inputs, dtype=tf.float16)

    # Several convolution blocks to distill a non-linear problem into a linear one
    x = conv_block(x, filters=32, kernel=7, strides=2)
    x = conv_block(x, filters=64, kernel=5, strides=1)
    x = conv_block(x, filters=64, kernel=3, strides=1)

    # Pool so we have less units going into the dense layer
    # Greatly reduces parameter count which lets us have bigger conv blocks
    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = Flatten()(x)

    # Dense layers to solve a linear problem
    x = Dense(units=512, activation="relu")(x)
    outputs = Dense(units=len(possible_actions), activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    model.summary()

    return model


def conv_block(x, filters, kernel, strides):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(0.001),
        activation="swish",
    )(x)
    x = BatchNormalization()(x)
    return x


def resize_frame(frame):
    frame = frame[30:-12, 5:-4]
    frame = np.average(frame, axis=2)
    frame = frame[..., np.newaxis]
    frame = tf.image.resize(frame, (84, 84))
    frame = frame[:, :, 0]
    return frame
