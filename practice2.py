# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# hyper parameters
x_range = 400
num_point = 100
num_iters = 10000 
num_dense_layer = 10
num_hidden_units = 100
learning_rate = 8e-5

#source data
ax = np.arange(0, x_range)
ay = 0.0
for i in range(1, 30, 2):
  ay += (np.sin(i*0.01*ax*np.pi) / i)
  
y = ay + (0.5 * np.random.rand(x_range))
indice = np.sort(np.random.choice(x_range, num_point))
x = ax[indice]
y = ay[indice]

#placeholder
x_placeholder = tf.placeholder(tf.float32, [None])
y_placeholder = tf.placeholder(tf.float32, [None])

# change shape to [batch_size, 1]
bx = tf.expand_dims(x_placeholder,axis=1)
by = tf.expand_dims(y_placeholder,axis=1)

#Full connect layer
output = bx
for i in range(num_dense_layer):
  output = tf.layers.dense(output, num_hidden_units, activation=tf.nn.tanh)
predict_y = tf.layers.dense(output, 1)

#loss function MAE
loss = tf.losses.absolute_difference(by, predict_y)

#minimize
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 10000 times
    for epoch in range(num_iters):

        output_x, output_y, output_predict, _ = sess.run(
                [bx, by, predict_y, train_op],
                 feed_dict={x_placeholder: x, y_placeholder: y})
        
        #print it
        if epoch % 1000 == 0:
            plt.figure('output')
            plt.clf()
            plt.plot(output_x, output_y, 'o')
            plt.plot(output_x, output_predict, '-')
            plt.show(block=False)
            plt.pause(0.01)
