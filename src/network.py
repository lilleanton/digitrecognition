from random import uniform
from copy import deepcopy as copy
import numpy as np


# Activation function for gradient descent, along with derivative
def activation(x):
    return 1 / (1 + np.exp(-x))  # Classic sigmoid


def d_activation(x):
    return activation(x) * (1 - activation(x))  # Derivative


# A single segment of a pack, used later in Network.back_propagation_activations()
class TrainingPiece:
    def __init__(self, data, label, net):
        self.data = data

        net.inputData(data)
        net.propagate()

        self.outputPrediction = net.output()
        self.outputTarget = [1 * (label is i) for i in range(10)]
        self.loss = [self.outputPrediction[i] - self.outputTarget[i]
                     for i in range(10)]  # Store loss output for backpropagation of the activations
        
        # Loss value, for keeping track of network performance
        self.total_loss = sum(self.loss[i] * self.loss[i] for i in range(10))


# Training set of fixed size for stochastic gradient descent
class TrainingPack:
    def __init__(self, data, labels, net):
        self.pieces = []  # All segments of the pack are stored in a TrainingPiece format for easy access
        self.net = copy(net)  # Store a copy of the network at the time of training
        self.pack_loss = 0  # Total pack loss statistic for all TrainingPiece

        for n, k in zip(data, labels):
            self.pieces.append(TrainingPiece(n, k, net))
        for i in self.pieces:
            self.pack_loss += i.total_loss
        self.pack_loss /= len(self.pieces)


class Network:
    def __init__(self):
        # Numpy matrices for highly optimized computations
        self.a = [np.matrix([[float(0)] for j in range(784)]),  # Input layer, 28 * 28 = 784 pixels in total
                  np.matrix([[float(0)] for j in range(16)]),   # Two 16 node layers, courtesy of Grant Sanderson
                  np.matrix([[float(0)] for j in range(16)]),   #
                  np.matrix([[float(0)] for j in range(10)])]   # Output layer, 10 nodes for digits 0-9

        # Caching layers before activations for backpropagation
        self.z = copy(self.a)
        
        # Weights
        self.w = [np.matrix([[uniform(-1, 1) for k in range(self.a[i - 1].shape[0])]
                             for j in range(self.a[i].shape[0])])
                  for i in range(1, len(self.a))]
        
        # Biases
        self.b = [np.matrix([[uniform(-1, 1)] for j in range(self.a[i].shape[0])]) for i in range(1, len(self.a))]

    def inputData(self, data):
        for idx, i in enumerate(data):
            self.a[0][idx, 0] = i

    def output(self):
        return [self.a[-1].item(i, 0) for i in range(self.a[-1].shape[0])]

    def propagate(self):
        for i in range(len(self.a) - 1):
            self.z[i + 1] = np.matmul(self.w[i], self.a[i]) - self.b[i]
            self.a[i + 1] = activation(self.z[i + 1])

    def back_propagate(self, pack):
        # Create temporary objects of the input space to generate a negative gradient
        new_w = copy(self.w)
        new_b = copy(self.b)

        for item in pack.pieces:
            # Get activation derivatives
            da = self.back_propagate_activations(item)
            step = 0.01  # Step constant, lower values are safer for finding a local minimum, albeit slower

            # Apply weight derivatives
            for l in range(len(self.w)):
                new_w[l] -= self.get_d_weight_layer(l, da) * step

            # Apply bias derivatives
            for l in range(len(self.b)):
                new_b[l] -= self.get_d_bias_layer(l, da) * step
                
        # Apply negative gradient
        self.w = copy(new_w)
        self.b = copy(new_b)
        
    # Weight derivatives for a layer 'l'
    def get_d_weight_layer(self, l, da):
        temp = np.zeros(self.w[l].shape)
        
        for a in range(temp.shape[0]):
            for b in range(temp.shape[1]):
                temp[a, b] = self.a[l].item(b, 0) * \
                             d_activation(self.z[l + 1][a, 0]) * \
                             da[l][a]

        return temp
    
    # Bias derivatives for a layer 'l'
    def get_d_bias_layer(self, l, da):
        temp = np.zeros(self.b[l].shape)

        for a in range(temp.shape[0]):
            temp[a][0] = d_activation(self.z[l + 1][a, 0]) * da[l][a]

        return temp
    
    # Get all activations derivatives to later compute the weights and biases
    def back_propagate_activations(self, piece):
        # Propagate
        self.inputData(piece.data)  # Piece of data with loss values
        self.propagate()

        # Layers of backpropagation
        da = []
        l = -1

        # Get first layer
        da.append([2 * piece.loss[a] for a in range(self.a[l].shape[0])])

        # Get remaining layers
        for i in range(len(self.a) - 2):
            da.insert(0, [sum([self.w[l][a, b] *
                               d_activation(self.z[l][a, 0]) *
                               da[l][a] for a in range(self.a[l].shape[0])]) for b in range(self.a[l - 1].shape[0])])
            l -= 1
        return da
