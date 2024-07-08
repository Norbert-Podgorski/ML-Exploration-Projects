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
        self.learning_rate = 0.1

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
        return 2 * (x - target) / x.size

