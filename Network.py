import numpy as np
from mnist import MNIST


class Network:
    def __init__(self, shape):
        self.shape = shape
        self.L = len(self.shape)
        self.weights = [np.random.rand(j, k)
                        for j, k in zip(self.shape[1:], self.shape[:-1])]
        self.biases = [np.random.rand(j, 1) for j in self.shape[1:]]

    def der_cost_func(self, a, y):
        return y-a

    def backprop(self, x, y):
        # init the grad matrices
        nabla_w = [np.zeros(s)
                   for s in zip(self.shape[1:], self.shape[:-1])]
        nabla_b = [np.zeros((j, 1)) for j in self.shape[1:]]

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
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def result(self, a):
        return np.argmax(a)

    def learn(self, x, y, eta):
        nabla_w, nabla_b = self.backprop(x, y)
        self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, data_set, labels, eta):
        for x, y in zip(data_set, labels):
            self.learn(x, y, eta)

    def test(self, data_set, labels):
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
    labels = [np.array([int(y == k) for k in range(10)])[:, None]
              for y in labels]
    timages = np.array(timages)
    timages = timages[:, :, None]
    tlabels = [np.array([int(y == k) for k in range(10)])[:, None]
               for y in tlabels]

    return images, labels, timages, tlabels


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def main():
    import Network as nt
    img, lab, timg, tlab = load_training_set()
    net = Network([784, 30, 10])
    net.train(img, lab, 3)

    s = net.test(timg, tlab)
    l = len(timg)
    r = s/l

    print("s : ", s)
    print("l : ", l)
    print("r : ", r)


if __name__ == "__main__":
    main()
