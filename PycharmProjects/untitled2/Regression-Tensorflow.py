import tensorflow as tf
import numpy  as np
import matplotlib.pyplot as plt

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)
    ##Platzhalter definieren
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_x')
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
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')

    ##Modell trainieren
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')

    ## Erzeugen zufälliger Beispielsdaten für die Regression

    np.random.seed(0)


    def make_random_data():
        x = np.random.uniform(low=-2, high=4, size=200)
        y = []
        for t in x:
            r = np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None)
            y.append(r)
        return x, 1.726 * x - 0.84 + np.array(y)
x, y = make_random_data()
plt.plot(x, y, 'o')
plt.show()

## Training/Test-Aufteilung
x_train, y_train = x[:100], y[:100]
x_test,  y_test  = x[100:], y[100:]
n_epochs = 500
training_costs =[]

with tf.Session(graph=g) as sess:
   # new_saver = tf.train.import_meta_graph('./trainiertes-Modell.meta')
    #new_saver.restore(sess, './trainiertes-Modell')

    ## Variablen initialisieren
    sess.run(tf.global_variables_initializer())

    ## Modell n_epochs lang trainieren
    for e in range(n_epochs):
        c, _ = sess.run(['cost:0', 'train_op'] , feed_dict={'tf_x:0': x_train, 'tf_y:0' :y_train})
        training_costs.append(c)
        if not e % 50 == 0:
            print('Epoche {:4d} : {:.4f}' .format(e,c))

plt.plot(training_costs)
plt.show()
