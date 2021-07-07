import numpy as np
from mnist import MNIST


class Network:
    def __init__(self, shape):
        self.shape = shape
        self.L = len(self.shape)
        self.weights = [np.zeros(s)
                        for s in zip(self.shape[1], self.shape[:-1])]
        self.biases = [np.zeros((j, 1)) for j in self.shape[1:]]

    def der_cost_func(self, a, y):
        return y-a

    def sigmoid_func(self, x):
        return 1/(1+np.exp(-x))

    def der_sigmoid_func(self, x):
        return self.sigmoid_func(x)*(self.sigmoid_func(x) - 1)

    def backprop(self, x, y):
        # init the grad matrices
        nabla_w = [np.zeros(s)
                   for s in zip(self.shape[1:], self.shape[:-1])]
        nabla_b = [np.zeros((j, 1)) for j in self.shape[1:]]

        # feedforward
        a = x
        acts = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.sigmoid_func(z)
            acts.append(a)

        # end of the Network
        delta = self.der_cost_func(a, y)*self.der_sigmoid_func(z)
        nabla_w[-1] = np.dot(delta, np.transpose(acts[-2]))
        nabla_b[-1] = delta

        # backprop
        for l in range(2, self.L - 1):
            delta = np.dot(np.transpose(
                self.weights[-l+1]), delta)*self.der_sigmoid_func(zs[-l])
            nabla_w[-l] = np.dot(delta, np.transpose(acts[-1-l]))
            nabla_b[-l] = delta

        return nabla_w, nabla_b

    def output(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid_func(z)
        return a

    def result(self, a):
        return np.argmax(a)

    def learn(self, x, y, eta):
        nabla_w, nabla_b = self.backprop(x, y)
        self.weights = self.weights - eta * nabla_w
        self.biases = self.biases - eta * nabla_b

    def train(self, data_set, labels, eta):
        for x, y in zip(data_set, labels):
            self.learn(x, y, eta)

    def test(self, data_set, labels, eta):
        c = 0
        for x, y in zip(data_set, labels):
            if self.result(x) == y:
                c += 1
        return c


def load_training_set():
    mndata = MNIST('./data')
    mndata.gz = True
    images, labels = mndata.load_training()
    timages, tlabels = mndata.load_testing()

    images = np.array(images)
    images = images[:, :, None]
    timages = np.array(timages)
    timages = timages[:, :, None]

    return images, labels, timages, tlabels


def main():
    return 0


if __name__ == "__main__":
    main()
