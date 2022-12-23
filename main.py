from random import uniform, shuffle
from copy import deepcopy as copy
import pickle
import draw
import numpy as np
import time
import mnist

def execute_t(func, arg):
    start_time = time.time()
    val = func(arg)
    print("--- %s seconds ---" % (time.time() - start_time))
    return val

def activation(x):
    return 1 / (1 + np.exp(-x))

def d_activation(x):
    return activation(x) * (1 - activation(x))


class TrainingPiece:
    def __init__(self, data, label, net):
        self.data = data

        net.inputData(data)
        net.propagate()

        self.outputPrediction = net.output()
        self.outputTarget = [1*(label is i) for i in range(10)]
        self.loss = [self.outputPrediction[i]-self.outputTarget[i]
                     for i in range(10)]

        self.total_loss = sum(self.loss[i]*self.loss[i] for i in range(10))


class TrainingPack:
    def __init__(self, data, labels, net):
        self.pieces = []
        self.net = copy(net)
        self.pack_loss = 0

        for n, k in zip(data, labels):
            self.pieces.append(TrainingPiece(n, k, net))
        for i in self.pieces:
            self.pack_loss += i.total_loss
        self.pack_loss /= len(self.pieces)


class Network:
    def __init__(self):
        self.a = [np.matrix([[float(0)] for j in range(784)]),
                  np.matrix([[float(0)] for j in range(16)]),
                  np.matrix([[float(0)] for j in range(16)]),
                  np.matrix([[float(0)] for j in range(10)])]
        self.z = copy(self.a)
        self.w = [np.matrix([[uniform(-1, 1) for k in range(self.a[i - 1].shape[0])]
                  for j in range(self.a[i].shape[0])])
                  for i in range(1, len(self.a))]
        self.b = [np.matrix([[uniform(-1, 1)] for j in range(self.a[i].shape[0])]) for i in range(1, len(self.a))]

    def inputData(self, data):
        for idx, i in enumerate(data):
            self.a[0][idx, 0] = i

    def output(self):
        return [self.a[-1].item(i, 0) for i in range(self.a[-1].shape[0])]

    def propagate(self):
        for i in range(len(self.a) - 1):
            self.z[i + 1] = np.matmul(self.w[i], self.a[i])-self.b[i]
            self.a[i + 1] = activation(self.z[i + 1])

    def back_propagate(self, pack):
        new_w = copy(self.w)
        new_b = copy(self.b)

        for item in pack.pieces:
            # Get derivatives
            da = self.back_propagate_activations(item)
            step = 0.1

            # Apply weight derivatives
            for l in range(len(self.w)):
                new_w[l] -= self.get_d_weight_layer(l, da) * step

            # Apply bias derivatives
            for l in range(len(self.b)):
                new_b[l] -= self.get_d_bias_layer(l, da) * step

        self.w = copy(new_w)
        self.b = copy(new_b)

    def get_d_weight_layer(self, l, da):
        temp = np.zeros(self.w[l].shape)

        for a in range(temp.shape[0]):
            for b in range(temp.shape[1]):
                temp[a, b] = self.a[l].item(b, 0) * \
                d_activation(self.z[l + 1][a, 0]) * \
                da[l][a]

        return temp

    def get_d_bias_layer(self, l, da):
        temp = np.zeros(self.b[l].shape)

        for a in range(temp.shape[0]):
            temp[a][0] = d_activation(self.z[l+1][a, 0]) * da[l][a]

        return temp

    def back_propagate_activations(self, pack):
        # Propagate
        self.inputData(pack.data)
        self.propagate()

        # Layers of backpropagation
        da = []
        l = -1

        # Get first layer
        da.append([2 * pack.loss[a] for a in range(self.a[l].shape[0])])

        # Get remaining layers
        for i in range(len(self.a)-2):
            da.insert(0, [sum([self.w[l][a, b] *
                               d_activation(self.z[l][a, 0]) *
                               da[l][a] for a in range(self.a[l].shape[0])]) for b in range(self.a[l - 1].shape[0])])
            l -= 1
        return da


def testNet(name, show_diagrams):
    data = pickle.load(open('testnormal.pkl', 'rb'))
    labels = pickle.load(open('testlabels.pkl', 'rb'))

    network = pickle.load(open(name, 'rb'))

    testing = list(zip(data, labels))
    shuffle(testing)

    num_correct = 1
    num_incorrect = 1

    for data, label in testing:
        network.inputData(data)
        network.propagate()

        guess = [index for index, item in enumerate(network.output()) if item == max(network.output())]
        if guess[0] == label:
            num_correct += 1
        else:
            num_incorrect += 1

        print([round(i, 2) for i in network.output()])
        print(f'{guess} - {"correct" if guess[0] == label else "incorrect"}')
        print(f'Ratio: {round(num_correct/(num_correct+num_incorrect), 3)}')
        print('\n')
        if show_diagrams:
            mnist.plot(data)


def trainNet(name=None):

    print('Loading training data...')
    data = np.array_split(pickle.load(open('trainingnormals.pkl', 'rb'))[0:60000], 600)
    labels = np.array_split(pickle.load(open('traininglabels.pkl', 'rb'))[0:60000], 600)

    for i in range(len(data)):
        data[i] = data[i].tolist()
    for i in range(len(labels)):
        labels[i] = labels[i].tolist()

    if name is None:
        tesla = Network()
    else:
        tesla = pickle.load(open(name, 'rb'))

    print(f'Commencing training with {len(data)} packs')
    for n, k in zip(data, labels):
        pack = TrainingPack(n, k, tesla)
        execute_t(tesla.back_propagate, pack)
        print(pack.pack_loss)
        if input('Exit? ') == 'y':
            break

    pickle.dump(tesla, open(input('What would you like to name the dump? '), 'wb'))


if __name__ == '__main__':
    draw.drawMode('lorentz008')