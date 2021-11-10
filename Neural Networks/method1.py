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
    model = tf.keras.Sequential(
        [
            Input(shape=(28, 28)),
            Reshape(target_shape=(28, 28, 1)),
            Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="swish",
                padding="valid",
                kernel_regularizer=l2(0.01),
                use_bias=False,
            ),
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="swish",
                padding="valid",
                kernel_regularizer=l2(0.01),
                use_bias=False,
            ),
            Flatten(),
            Dense(units=64, activation="swish"),
            Dense(units=32, activation="swish"),
            Dense(units=10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(lr=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
