# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# hyper parameters
x_range = 720
num_point = 100
num_iters = 10000 
num_dense_layer = 10
num_hidden_units = 100
learning_rate = 8e-5

#source data
ax = np.arange(0, x_range)
ay = 0.0
for i in range(1, 30, 2):
  ay += (np.sin(i*ax*(np.pi/180.0)) / i)
  
y = ay + (0.5 * np.random.rand(x_range))
indice = np.sort(np.random.choice(x_range, num_point))
x = ax[indice]
y = ay[indice]

#placeholder
time_placeholder = tf.placeholder(tf.float32, [None])
value_placeholder = tf.placeholder(tf.float32, [None])

# change shape to [batch_size, 1]
time = tf.expand_dims(time_placeholder,axis=1)
value = tf.expand_dims(value_placeholder,axis=1)

#Full connect layer
output = time
for i in range(num_dense_layer):
  output = tf.layers.dense(output, num_hidden_units, activation=tf.nn.tanh)
predict_value = tf.layers.dense(output, 1)

#loss function MAE
loss = tf.losses.absolute_difference(value, predict_value)

#minimize
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 10000 times
    for epoch in range(num_iters):

        output_time, output_value, output_predict, _ = sess.run(
                [time, value, predict_value, train_op],
                 feed_dict={time_placeholder: x, value_placeholder: y})
        
        #print it
        if epoch % 1000 == 0:
            plt.figure()
            plt.plot(output_time, output_value, 'o')
            plt.plot(output_time, output_predict, '-')
            plt.show()
