import tensorflow as tf
import numpy  as np
import matplotlib.pyplot as plt

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)
    ##Platzhalter definieren
    tf_x = tf.placeholder(shape=(None), dtype= tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_y')

    ##Variablen definieren
    weight = tf.Variable(
        tf.random_normal(
            shape=(1, 1),
            stddev=0.25),
            name='weight')
    bias = tf.Variable(0.0, name='bias')


    ##Modell erstellen
    y_hat = tf.add(weight * tf_x, bias, name='y_hat')

    ##Straffunktion berechnen
    cost = tf.reduce_mean(tf.square(tf_y- y_hat), name= 'cost')

    ##Modell trainieren
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')

## Erzeugen zufälliger Beispielsdaten für die Regression

    np.random.seed(0)

    def make_random_data():
        x = np.random.uniform(low= -2, high= 4, size=200)
        y= []
        for t in x:
            r = np.random.normal(loc= 0.0, scale=(0.5 + t*t/3), size=None)
            y.append(r)
    return x, 1.726 * x -0.84 + np.array(y)
x, y = make_random_data()
plt.plot(x, y, 'o')
plt.show()