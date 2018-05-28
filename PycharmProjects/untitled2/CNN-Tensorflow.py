import os
import struct
import numpy as np
import tensorflow as tf


def load_mnist(path, kind='train'):
    """MNIST-Daten von path laden"""
    label_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


##Datein einlesen
X_data, y_data = load_mnist('MNIST_data', kind='train')
print('Zeilen: {}, Spalten:{}'.format(X_data.shape[0], X_data.shape[1]))

X_test, y_test = load_mnist('MNIST_data', kind='t10k')
print('Zeilen: {}, Spalten: {}'.format(X_test.shape[0], X_test.shape[1]))

X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]

print('Training:      ', X_train.shape, y_train.shape)
print('Validierung:   ', X_valid.shape, y_valid.shape)
print('Testdaten:     ', X_test.shape, y_test.shape)


## Mini-Batches
def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])

mean_vals = np.mean(X_train, axis=0)
std_val   = np.std(X_train)
X_train_centered =(X_train - mean_vals)/ std_val
X_valid_centered =(X_valid - mean_vals)/ std_val
X_test_centered =(X_test - mean_vals)/ std_val
