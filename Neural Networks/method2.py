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


def build_model():
    input_layer = Input(shape=(28, 28))
    reshape_layer = Reshape(target_shape=(28, 28, 1))

    conv_layer_1 = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="swish",
        padding="valid",
        kernel_regularizer=l2(0.01),
        use_bias=False,
    )
    conv_layer_2 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="swish",
        padding="valid",
        kernel_regularizer=l2(0.01),
        use_bias=False,
    )

    flatten_layer = Flatten()

    dense_layer_1 = Dense(units=64, activation="swish")
    dense_layer_2 = Dense(units=32, activation="swish")

    output_layer = Dense(units=10, activation="softmax")

    model = tf.keras.Sequential(
        [
            input_layer,
            reshape_layer,
            conv_layer_1,
            conv_layer_2,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            output_layer,
        ]
    )

    model.compile(
        optimizer=Adam(lr=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
