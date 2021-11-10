import method1
import method2
import method3
import method4
import tensortorch_model

# import pytorch_model

import tensorflow as tf


def tf_training():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    model = method1.build_model()
    model.summary()

    model.fit(
        x=train_images,
        y=train_labels,
        validation_data=(test_images, test_labels),
        batch_size=128,
        epochs=10,
    )


def tensortorch_training():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = fashion_mnist.load_data()
    train_images = train_images / 1.0
    test_images = test_images / 1.0

    model = tensortorch_model.CNN()

    model.fit(
        x=train_images,
        y=train_labels,
        test_x=test_images,
        test_y=test_labels,
        batch_size=128,
        epochs=10,
    )


def pytorch_training():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    model = pytorch_model.CNN()
    print(model.model)
    model.fit(
        x=train_images,
        y=train_labels,
        test_x=test_images,
        test_y=test_labels,
        batch_size=128,
        epochs=10,
    )


if __name__ == "__main__":
    tf_training()
    # tensortorch_training()
    # pytorch_training()
