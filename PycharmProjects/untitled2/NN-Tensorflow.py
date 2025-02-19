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

X_train, y_train= load_mnist('MNIST_data', kind='train')
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

import tensorflow as tf


n_features = X_train_centered.shape[1]
n_classes  = 10
random_seed = 123

np.random.seed(random_seed)
g = tf.Graph()
with g.as_default():

    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None , n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')
    h1 = tf.layers.dense(input= tf_x, units = 50, activation= tf.tanh, name = 'layer1')
    h2 = tf.layers.dense(input= tf_x, units = 50, activation= tf.tanh, name = 'layer2')
    logits = tf.layers.dense(input= h2, units= 10, activation= None, name= 'layer3')

    predictions = {
        'classes' : tf.argmax(logits, axis=1, name= 'predicted_classes'),
        'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')
    }

##Straffunktion und Optimierer definieren:

with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels= y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)
    init_op  = tf.global_variables_initializer()

def create_batch_generator(X, y, batch_size= 128, shuffle= False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data: np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield  (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])


## Sitzung zum Starten des Graphen erzeugen
sess = tf.Session(graph=g)
## Operator zur Variableninitialisierung ausführen
sess.run(init_op)

## 50 Trainingsepochen:
for epoch in range(50):
    training_cost = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size= 64)

    for batch_X, batch_y in batch_generator:
    ## Dictionary für die Datenübergabe vorbereiten:
    feed = {tf_x:batch_X, tf_y:batch_y}
    _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
    training_costs.append(batch_cost)
    print('-- Epoche %2d' 'Duchschnittswert der Straffunktion: %.4f' %(epoch+1, np.mean(training_costs)
    ))

##Vorhersagen für die Testdatenmenge treffen:
feed = {tf_x : X_test_centered}
y_pred = sess.run(predictions['classes'], feed_dict=feed)
print('Korrektklassifizierungsrate Test: %2f%%' %(100* np.sum(y_pred == y_test) /y_test.shape[0]))