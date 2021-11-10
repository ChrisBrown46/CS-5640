import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.keras.regularizers import l2


class CnnModel(tf.keras.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.reshape = nn.Reshape(target_shape=(28, 28, 1))
        self.flatten = nn.Flatten()
        self.batch_norm1 = nn.BatchNormalization()
        self.batch_norm2 = nn.BatchNormalization()

        self.conv1 = nn.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="swish",
            padding="valid",
            kernel_regularizer=l2(0.01),
            use_bias=False,
        )
        self.conv2 = nn.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="swish",
            padding="valid",
            kernel_regularizer=l2(0.01),
            use_bias=False,
        )

        self.fc1 = nn.Dense(units=64, activation="swish")
        self.fc2 = nn.Dense(units=32, activation="swish")
        self.fc3 = nn.Dense(units=10, activation="softmax")

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNN(object):
    def __init__(self):
        self.model = CnnModel()
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    def fit(self, x, y, test_x, test_y, batch_size=4, epochs=10):
        """
        First, prepare the data so TensorFlow can use it
        Second, iterate over the training set
            Third, clear the old gradients
            Fourth, compute the forward pass and backprop the error
        Fifth, iterate over the test set to get the accuracy
        Sixth, print the loss
        """

        train_dataset, test_dataset = self.prepare_data(
            x, y, test_x, test_y, batch_size
        )

        for epoch in range(epochs):
            train_loss, test_loss = 0.0, 0.0

            for inputs, labels in train_dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(inputs, training=True)
                    loss = self.criterion(labels, logits)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

                train_loss += loss.numpy()

            for inputs, labels in test_dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(inputs, training=False)
                    loss = self.criterion(labels, logits)

                test_loss += loss.numpy()

            print(
                f"Iteration: {epoch}\tTrainLoss: {train_loss:.4f}\tTestLoss: {test_loss:.4f}"
            )

    def prepare_data(self, x, y, test_x, test_y, batch_size):
        """
        First, convert the data to a dataset
        Second, shuffle the dataset
        Third, create the batch size
        Fourth, create the dataset iterator
        """
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=batch_size * 2)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_dataset, test_dataset
