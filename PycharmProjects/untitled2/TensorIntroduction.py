import tensorflow as tf
import numpy as np

g = tf.Graph()

#Platzhalter definieren
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')

r1= tf_a-tf_b
r2= r1*2
z = r2+tf_c

with tf.Session(graph=g) as sess:
    feed = {tf_a:1,tf_b:2,tf_c:3
    }
    print('z', sess.run(z, feed_dict= feed))

#Variablen definieren
g1 = tf.Graph()

with g1.as_default():
     w = tf.Variable(np.array([[1,2,3,4],
                              [5,6,7,8]]), name= 'w')
print(w)

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))



g2 = tf.Graph()

with g2.as_default():
     w1 = tf.Variable(1, name= 'w1')
     w2 = tf.Variable(2, name = 'w2')
     init_op = tf.global_variables_initializer()


with tf.Session(graph=g2) as sess:
   sess.run(init_op)
   print('w1:', sess.run(w1))
   print('w2:', sess.run(w2))

#Wiederverwendung von Variablen

################################
####### Hilfsfunktionen ########
################################

def build_clasifier(data, labels, n_classes =2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name= 'weights', shape=(data_shape[1], n_classes), dtype=tf.float32)
    bias    = tf.get_variable(name= 'bias', initializer= tf.zeros(shape=n_classes))
    logits  = tf.add(tf.matmul(data, weights), bias, name= 'logits')
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape.as_list()
    w1 = tf.Variable(
        tf.random_normal(shape=(data_shape[1], n_hidden)), name= 'w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden), name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')

    w2 = tf.Variable(
        tf.random_normal(shape=(n_hidden, data_shape[1])), name= 'w2')

    b2 = tf.Variable(tf.zeros(shape=data_shape[1]), name='b2')

    output = tf.add(tf.matmul(hidden, w2,), b2, name='output')
    return output, tf.nn.sigmoid(output)
    )

#############################
### Graphen erstellen #######
#############################

batch_size = 64
g= tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype= tf.float32, name='tf_x')


    ##Generator erstellen
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data= tf_X, n_hidden = 50)

    ##Klassifizierer erstellen
        with tf.variable_scope('classifier') as scope:
            ##Klassifizierer für die ursprünglichen Daten:
            clas_out1 = build_clasifier(data= tf_X, labels=tf.ones(shape=batch_size))

        ## Wiederverwenden des Klassifizierers für die generierten Daten
            scope.reuse_variables()
            cls_out2 = build_clasifier(data= gen_out1[1], labels=tf.zeros(shape=batch_size))

