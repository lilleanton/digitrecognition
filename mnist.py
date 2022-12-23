import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot(img):
    data = np.array([[img[i*28+j] for j in range(28)] for i in range(28)])
    plt.imshow(data, interpolation='none', cmap='gray', vmin=0, vmax=1)
    plt.show()


def importImages():
    imgs_binary = []
    f = open('data/test-images.dat', 'rb')
    print(f.read(16))
    for i in range(10000):
        temp = []
        for j in range(784):
            temp.append(int.from_bytes(f.read(1), byteorder='big'))
        imgs_binary.append(temp)
    f.close()

    imgs_normalized = []
    for idx, i in enumerate(imgs_binary):
        print(idx)
        imgs_normalized.append([j / 255 for j in i])

    pickle.dump(imgs_normalized, open('testnormal.pkl', 'wb'))


def importLabels():
    imgs_binary = []
    f = open('data/test-labels.dat', 'rb')
    print(f.read(8))

    labels = []
    for i in range(10000):
        labels.append(int.from_bytes(f.read(1), byteorder='big'))
    f.close()

    pickle.dump(labels, open('testlabels.pkl', 'wb'))
