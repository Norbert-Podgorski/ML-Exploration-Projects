import random

import numpy as np


class Layer:
    def forward(self, x):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, x, grad):
        pass


class Linear(Layer):
    def __init__(self, in_dims, out_dims):
        self.params = np.random.normal(0, 0.1, size=[out_dims, in_dims])
        self.learning_rate = 0.01

    def forward(self, x):
        return self.params @ x

    def backward(self, x, grad):
        params_grad = grad @ x.T
        self.update_params(params_grad)
        return self.params.T @ grad

    def update_params(self, params_grad):
        self.params -= params_grad * self.learning_rate


class Relu(Layer):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x, grad):
        return grad * (x > 0).astype(np.float32)


class MSELoss:
    def forward(self, x, target):
        return np.square(x - target).mean()

    def backward(self, x, target):
        return 2 * (x - target) / target.size


class Model:
    def __init__(self):
        self.layers = [
            Linear(2, 10),
            Relu(),
            Linear(10, 1)
        ]
        self.cost = MSELoss()

    def forward_step(self, x):
        inputs = [x]
        for layer in self.layers:
            x = layer(x)
            inputs.append(x)
        return x, inputs

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward_step(self, inputs, target):
        grad = self.cost.backward(inputs[-1], target)
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(inputs[i], grad)

    def train_step(self, x, y):
        y_pred, inputs = self.forward_step(x)
        loss = self.cost.forward(y_pred, y)
        self.backward_step(inputs, y)
        return loss

    def train(self, train_x, train_y):
        losses = []
        for i in range(10000):
            idx = random.randrange(train_x.shape[0])
            x = train_x[idx][..., np.newaxis]
            y = train_y[idx][..., np.newaxis]
            loss = self.train_step(x, y)
            losses.append(loss)
            print(np.mean(losses[-50:]))


train_x = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

train_y = np.array(
    [
        [0],
        [1],
        [1],
        [0]
    ]
)

model = Model()
model.train(train_x, train_y)

print("\nPredictions:")
print(model(np.array([[0], [0]])))
print(model(np.array([[0], [1]])))
print(model(np.array([[1], [0]])))
print(model(np.array([[1], [1]])))
