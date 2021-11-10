import tensorflow as tf
from tensorflow.keras.layers import (
    Reshape,
    Conv2D,
    Dense,
    Input,
    Flatten,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_conv_block(input_layer):
    conv = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="swish",
        kernel_regularizer=l2(0.01),
        use_bias=False,
    )(input_layer)
    conv = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="swish",
        kernel_regularizer=l2(0.01),
        use_bias=False,
    )(conv)

    return conv


def build_dense_block(input_layer):
    dense = Dense(units=64, activation="swish")(input_layer)
    dense = Dense(units=32, activation="swish")(dense)

    return dense


def build_model():
    input_layer = Input(shape=(28, 28))
    hidden_layers = Reshape(target_shape=(28, 28, 1))(input_layer)

    hidden_layers = build_conv_block(hidden_layers)
    hidden_layers = Flatten()(hidden_layers)
    hidden_layers = build_dense_block(hidden_layers)

    output_layer = Dense(units=10, activation="softmax", kernel_regularizer=l2(0.01))(
        hidden_layers
    )

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(lr=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
