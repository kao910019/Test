# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#source data
t = np.arange(0, 360)
y = np.sin(t*(np.pi/180))

#placeholder
time_placeholder = tf.placeholder(tf.float32, [None])
value_placeholder = tf.placeholder(tf.float32, [None])

# change shape to [batch_size, 1]
time = tf.expand_dims(time_placeholder,axis=1)
value = tf.expand_dims(value_placeholder,axis=1)

#Full connect layer
full_connect1 = tf.layers.dense(time, 50, activation=tf.nn.leaky_relu)
full_connect2 = tf.layers.dense(full_connect1, 25, activation=tf.nn.leaky_relu)
full_connect3 = tf.layers.dense(full_connect2, 50, activation=tf.nn.leaky_relu)
full_connect4 = tf.layers.dense(full_connect3, 25, activation=tf.nn.leaky_relu)
predict_value = tf.layers.dense(full_connect4, 1, activation=tf.nn.leaky_relu)

#loss function MAE
loss = tf.losses.absolute_difference(value, predict_value)

#minimize
optimizer = tf.train.AdamOptimizer(8e-4)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 2000 times
    for epoch in range(10000):
        output_time, output_value, output_predict, _ = sess.run(
                [time, value, predict_value, train_op],
                 feed_dict={time_placeholder: t, value_placeholder: y})
        
        #print it
        if epoch % 1000 == 0:
            plt.figure()
            plt.plot(output_time, output_value)
            plt.plot(output_time, output_predict)
            plt.show()







