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

    def cost_func(self, data, lbd=0):
        co = sum([np.linalg.norm(y-self.output(x))**2 for x, y in data])
        if lbd != 0:
            reg = lbd * sum([sum(sq) for sq in [x*x for x in self.weights]])
        else:
            reg = 0
        return (co + reg)/(2*len(data))

    def der_cost_func(self, a, y):
        return a-y

    def cross_entropy(self, data):
        b = []
        for x, y in data:
            a = self.output(x)
            b.append(y*np.log(a) + (1 - y) * np.log(1 - a))
        return -sum(b)/len(data)

    def der_cross_entropy(self, a, y):
        return (a - y)/(a*(1 - a))

    def backprop(self, x, y, der_fun):
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
        delta = der_fun(acts[-1], y)*der_sigmoid(zs[-1])
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

    def learn(self, mini_batch, eta, der_fun, lbd, n=1):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y, der_fun)
            nabla_w = [nw*(1 - eta*lbd/n) + dnw for nw,
                       dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        m = len(mini_batch)
        self.weights = [w - (eta / m) * nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b,
                       nb in zip(self.biases, nabla_b)]

    def train(self, data_set, eta, m, ep, fun, der_fun, lbd=0, verb=False):
        if verb:
            test_set = data_set[-10000:]
            data_set = data_set[:50000]

        n = len(data_set)
        for e in range(ep):
            random.shuffle(data_set)
            for k in range(n//m):
                self.learn(data_set[k*m:(k+1)*m], eta, der_fun, lbd, n)

            if verb:
                print("Ep", e+1, ":", self.test(test_set), "/", len(test_set))
                print("Loss:", fun(test_set))
            else:
                print("Ep", e+1, "...")

    def test(self, data_set):
        return sum([int(self.result(x) == np.argmax(y)) for x, y in data_set])


def load_training_set(fnorm):
    mndata = MNIST('./data')
    mndata.gz = True
    images, labels = mndata.load_training()
    timages, tlabels = mndata.load_testing()

    images = np.array(images)/fnorm
    images = images[:, :, None]
    labels = [np.array([int(y == k) for k in range(10)])[:, None]
              for y in labels]
    timages = np.array(timages)/fnorm
    timages = timages[:, :, None]
    tlabels = [np.array([int(y == k) for k in range(10)])[:, None]
               for y in tlabels]

    return list(zip(images, labels)), list(zip(timages, tlabels))


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def main():
    data, test = load_training_set(100)
    net = Network([784, 30, 10])

    eta_cross_entropy = 0.5
    mini_batch_size = 10
    epoch = 30
    lbd = 5

    print("train with:")
    print("eta:", eta_cross_entropy)
    print("mini_batch_size:", mini_batch_size)
    print("epoch:", epoch)
    print("lambda:", lbd, "\n")
    net.train(data, eta_cross_entropy, mini_batch_size, epoch,
              net.cross_entropy, net.der_cross_entropy, lbd, True)
    # net.train(data, eta, mini_batch_size, epoch,
    #          net.cost_func, net.der_cost_func, True)

    s = net.test(test)
    l = len(test)
    r = s/l

    print("s : ", s)
    print("l : ", l)
    print("r : ", r)


if __name__ == "__main__":
    main()
