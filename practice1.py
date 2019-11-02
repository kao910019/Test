# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#source data
x = np.arange(0, 720)
y = np.sin(x*(np.pi/180))

#placeholder
x_placeholder = tf.placeholder(tf.float32, [None])
y_placeholder = tf.placeholder(tf.float32, [None])

# change shape to [batch_size, 1]
bx = tf.expand_dims(x_placeholder,axis=1)
by = tf.expand_dims(y_placeholder,axis=1)

#Full connect layer
full_connect1 = tf.layers.dense(bx, 50, activation=tf.nn.tanh)
full_connect2 = tf.layers.dense(full_connect1, 25, activation=tf.nn.tanh)
full_connect3 = tf.layers.dense(full_connect2, 50, activation=tf.nn.tanh)
full_connect4 = tf.layers.dense(full_connect3, 25, activation=tf.nn.tanh)
predict_y = tf.layers.dense(full_connect4, 1, activation=tf.nn.tanh)

#loss function MAE
loss = tf.losses.absolute_difference(by, predict_y)

#minimize
optimizer = tf.train.AdamOptimizer(8e-4)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 10000 times
    for epoch in range(20000):
        output_x, output_y, output_predict, _ = sess.run(
                [bx, by, predict_y, train_op],
                 feed_dict={x_placeholder: x, y_placeholder: y})
        
        #print it
        if epoch % 1000 == 0:
            plt.figure()
            plt.plot(output_x, output_y)
            plt.plot(output_x, output_predict)
            plt.show()
