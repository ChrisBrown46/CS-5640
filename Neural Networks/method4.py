import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Reshape,
    Conv2D,
    Dense,
    Input,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_conv_block(input_layer, filters, kernel_size, strides):
    conv = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        activation="swish",
        kernel_regularizer=l2(0.01),
        use_bias=False,
    )(input_layer)

    return BatchNormalization()(conv)


def build_dense_block(input_layer, units):
    dense = Dense(units=units, activation="swish")(input_layer)
    # batch norm? dropout? l2 regularization?
    return dense


def build_head(input_layer):
    conv = build_conv_block(input_layer, filters=32, kernel_size=(3, 3), strides=(1, 1))
    conv = build_conv_block(conv, filters=64, kernel_size=(3, 3), strides=(1, 1))
    output = Flatten()(conv)

    return output


def build_tail(input_layer):
    dense = build_dense_block(input_layer, units=64)
    dense = build_dense_block(dense, units=32)
    output = Dense(units=10, activation="softmax")(dense)

    return output


def build_model():
    input_layer = Input(shape=(28, 28))
    reshape = Reshape(target_shape=(28, 28, 1))(input_layer)

    head = build_head(reshape)
    tail = build_tail(head)

    # head and tail are seperated so we can swap out the tail for a new one
    #   whenever we want, without having to retrain the whole model
    #   this method is called "transfer learning"
    # this also lets you make an arbitrary number of heads that all connect
    #   to the same tail
    # conversely, you can make an arbitrary number of tails that all connect
    #   to the same head
    head_model = Model(inputs=input_layer, outputs=head)
    tail_model = Model(inputs=input_layer, outputs=tail)

    tail_model.compile(
        optimizer=Adam(lr=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return head_model, tail_model
