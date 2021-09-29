import autograd
import autograd.numpy as np
from autograd.misc.optimizers import unflatten_optimizer


class Functions(object):
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def categorical_crossentropy(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        return -(targets * np.log(predictions + 1e-9)) / predictions.shape[0]

    @staticmethod
    def mean_squared_error(predictions, targets):
        return predictions - targets  # np.mean(np.abs(...))


class NeuralNetwork(object):
    def __init__(self):
        self.architecture = []
        self.parameters = []

    def add_layer(self, units):
        self.architecture.append(units)

        if len(self.architecture) > 1:
            self._initialize_weights()

    # Change the activation function in here
    # Allow params to be passed in: for AutoGrad use
    def forward_propogation(self, inputs, params=None):
        if params is None:
            params = self.parameters

        for weights, bias in params[:-1]:
            outputs = np.dot(inputs, weights) + bias
            inputs = Functions.relu(outputs)

        weights, bias = params[-1]
        outputs = np.dot(inputs, weights) + bias

        return outputs

    def fit(self, inputs, targets, learning_rate=1e-4, epochs=150):
        def objective(parameters, _):
            predictions = self.forward_propogation(inputs, parameters)
            return (np.square(targets - predictions)).mean()

        objective_grad = autograd.grad(objective)

        @unflatten_optimizer
        def adam(grad, x, _):
            b1, b2, eps = 0.9, 0.999, 10 ** -8
            m = np.zeros(len(x))
            v = np.zeros(len(x))
            for i in range(epochs):
                gradients = grad(x, i)
                m = (1 - b1) * gradients + b1 * m  # First  moment estimate.
                v = (1 - b2) * (gradients ** 2) + b2 * v  # Second moment estimate.
                mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
                vhat = v / (1 - b2 ** (i + 1))
                x += -learning_rate * mhat / (np.sqrt(vhat) + eps)
            return x

        self.parameters = adam(objective_grad, self.parameters)

    def _initialize_weights(self):
        input_dims = self.architecture[-2]
        output_dims = self.architecture[-1]

        scale = 1 / max(1.0, (input_dims + output_dims) / 2.0)
        limit = np.sqrt(3.0 * scale)
        weights = np.random.uniform(-limit, limit, size=(input_dims, output_dims))
        bias = np.random.uniform(-limit, limit, size=(output_dims))

        self.parameters.append((weights, bias))


# Build and test a MNIST network
if __name__ == "__main__":

    model = NeuralNetwork()
    model.add_layer(units=2)
    model.add_layer(units=50)
    model.add_layer(units=50)
    model.add_layer(units=3)

    epochs = 5000
    x = np.random.uniform(-2, 2, 2)
    y = np.random.uniform(-2, 2, 3)
    x = np.array([x] * epochs)
    y = np.array([y] * epochs)

    model.fit(x, y, learning_rate=0.001)

    prediction = model.forward_propogation(x[0])
    print(f"Expectation: {y[0]} | Actual: {prediction}")
