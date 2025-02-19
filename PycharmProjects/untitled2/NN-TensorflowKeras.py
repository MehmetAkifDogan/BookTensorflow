## Daten einlesen
import os
import struct
import numpy as np



def load_mnist(path, kind = 'train'):
    """MNIST-Daten von path laden"""
    label_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))

        labels = np.fromfile(lbpath, dtype= np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)

    return images, labels

X_train, y_train = load_mnist('MNIST_data', kind='train')
print('Zeilen: %d, Spalten: %d' %(X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('MNIST_data', kind = 't10k')
print('Zeilen: %d, Spalten: %d' %(X_test.shape[0], X_test.shape[1]))


##Zentrierung um Mittelwert und Normierung
mean_vals = np.mean(X_train, axis=0)
std_val  = np.std(X_train)

X_train_centered = (X_train - mean_vals)/ std_val
X_test_centered = (X_test - mean_vals)/ std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

#------------------------------------------------------------------------------------------
#---------------------NN with Keras and Tensorflow-----------------------------------------
#------------------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.keras as keras
np.random.seed(123)
tf.set_random_seed(123)

y_train_onehot = keras.utils.to_categorical(y_train)
print('Die ersten 3 Klassenbezeichnungen: ', y_train[:3])
print('\nDie ersten 3 Klassenbezeichnungen (OneHot): \n', y_train_onehot[:3])

#Feed Forward Neuronales Netz
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
#Training vom Neuronalen Netz
history = model.fit(X_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)
#Vorhersagen generieren
y_train_pred = model.predict_classes(X_train_centered,verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

print('Die ersten 3 Vorhersagen: ', y_train_pred[:3])
print('Korrektklassifizierungsrate Training: %.2f%%' %(train_acc*100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds  = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Korrektklassifizierungsrate Test: %.2f%%' %(test_acc*100))
