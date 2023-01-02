import numpy as np
import matplotlib.pyplot as plt
import pickle
from network import Network, TrainingPack
from random import shuffle


# Use pyplot for visualizing mnist data
def plot(img):
    data = np.array([[img[i*28+j] for j in range(28)] for i in range(28)])
    plt.imshow(data, interpolation='none', cmap='gray', vmin=0, vmax=1)
    plt.show()


# Import images
def importImages(path, n_img):
    imgs_binary = []
    f = open(path, 'rb')
    print(f.read(16))
    for i in range(n_img):
        temp = []
        for j in range(784):
            temp.append(int.from_bytes(f.read(1), byteorder='big'))
        imgs_binary.append(temp)
    f.close()

    imgs_normalized = []
    for idx, i in enumerate(imgs_binary):
        print(idx)
        imgs_normalized.append([j / 255 for j in i])

    pickle.dump(imgs_normalized, open(input('Enter new dump name: '), 'wb'))


# Import labels for images
def importLabels(path):
    imgs_binary = []
    f = open(path, 'rb')
    print(f.read(8))

    labels = []
    for i in range(10000):
        labels.append(int.from_bytes(f.read(1), byteorder='big'))
    f.close()

    pickle.dump(labels, open(input('Enter new dump name: '), 'wb'))


# Test a net's performance against a set
def testNet(name, show_diagrams, test_normals='testnormals.pkl', test_labels='testlabels.pkl'):
    # Import data
    data = pickle.load(open(test_normals, 'rb'))
    labels = pickle.load(open(test_labels, 'rb'))
    testing = list(zip(data, labels))
    shuffle(testing)

    # Import network
    network = pickle.load(open(name, 'rb'))

    num_correct = 1
    num_incorrect = 1

    # Test network
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
            plot(data)


# Training a net with provided data
def trainNet(name=None):
    # Import data
    print('Loading training data...')
    data = np.array_split(pickle.load(open('trainingnormals.pkl', 'rb'))[0:60000], 600)
    labels = np.array_split(pickle.load(open('traininglabels.pkl', 'rb'))[0:60000], 600)

    for i in range(len(data)):
        data[i] = data[i].tolist()
    for i in range(len(labels)):
        labels[i] = labels[i].tolist()

    # Create new network
    if name is None:
        tesla = Network()
    else:
        tesla = pickle.load(open(name, 'rb'))

    # Training
    print(f'Commencing training with {len(data)} packs')
    for n, k in zip(data, labels):
        pack = TrainingPack(n, k, tesla)  # Generate set of 100 training examples
        tesla.back_propagate(pack)  # Apply gradient descent
        print(pack.pack_loss)

    pickle.dump(tesla, open(input('What would you like to name the dump? '), 'wb'))
