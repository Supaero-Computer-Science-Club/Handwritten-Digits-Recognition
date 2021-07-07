import numpy as np
from mnist import MNIST
import random


class Network:
    def __init__(self, shape):
        self.shape = shape
        self.L = len(self.shape)
        self.weights = [np.random.randn(j, k)
                        for j, k in zip(shape[1:], shape[:-1])]
        self.biases = [np.random.randn(j, 1) for j in self.shape[1:]]

    def der_cost_func(self, a, y):
        return y-a

    def backprop(self, x, y):
        # init the grad matrices
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        a = x
        acts = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            acts.append(a)

        # end of the Network
        delta = self.der_cost_func(acts[-1], y)*der_sigmoid(zs[-1])
        nabla_w[-1] = np.dot(delta, acts[-2].transpose())
        nabla_b[-1] = delta

        # backprop
        for l in range(2, self.L):
            delta = np.dot(
                self.weights[-l+1].transpose(), delta) * der_sigmoid(zs[-l])
            nabla_w[-l] = np.dot(delta, acts[-1-l].transpose())
            nabla_b[-l] = delta

        return nabla_w, nabla_b

    def output(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def result(self, x):
        return np.argmax(self.output(x))

    def learn(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        m = len(mini_batch)
        self.weights = [w - eta / m * nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta/m*nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, data_set, eta, m, ep):
        len_mini_batch = len(data_set)//m
        for _ in range(ep):
            random.shuffle(data_set)
            for k in range(m):
                self.learn(
                    data_set[k*len_mini_batch:(k+1)*len_mini_batch], eta)

    def test(self, data_set):
        return sum([int(self.result(x) == np.argmax(y)) for x, y in data_set])


def load_training_set():
    mndata = MNIST('./data')
    mndata.gz = True
    images, labels = mndata.load_training()
    timages, tlabels = mndata.load_testing()

    images = np.array(images)
    images = images[:, :, None]
    labels = [np.array([int(y == k) for k in range(10)])[:, None]
              for y in labels]
    timages = np.array(timages)
    timages = timages[:, :, None]
    tlabels = [np.array([int(y == k) for k in range(10)])[:, None]
               for y in tlabels]

    return list(zip(images, labels)), list(zip(timages, tlabels))


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def main():
    data, test = load_training_set()
    net = Network([784, 30, 10])
    net.train(data, 3, 10, 30)

    s = net.test(test)
    l = len(test)
    r = s/l

    print("s : ", s)
    print("l : ", l)
    print("r : ", r)


if __name__ == "__main__":
    main()
