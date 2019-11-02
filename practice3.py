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
x_placeholder = tf.placeholder(tf.float32, [None])
y_placeholder = tf.placeholder(tf.float32, [None])

def Network(x, y):
    with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
        global_step = tf.Variable(0, name = 'global_step', trainable=False)
    
        # change shape to [batch_size, 1]
        x = tf.expand_dims(x,axis=1)
        y = tf.expand_dims(y,axis=1)
        
        #Full connect layer
        output = x
        for i in range(num_dense_layer):
          output = tf.layers.dense(output, num_hidden_units, activation=tf.nn.tanh)
        predict_y = tf.layers.dense(output, 1)
        
        #loss function MAE
        loss = tf.losses.absolute_difference(y, predict_y)
        
        #minimize
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        
        summary = tf.summary.merge([tf.summary.scalar("loss", loss)])
        
    return x, y, predict_y, summary, global_step, train_op


with tf.Session() as sess:
    with sess.graph.as_default():
        # Draw your neural network here.
        output = Network(x_placeholder, y_placeholder)
        
    # tensorboard writer
    summary_writer = tf.summary.FileWriter("tensorboard", sess.graph)
    
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 10000 times
    for epoch in range(num_iters):

        output_x, output_y, output_predict, output_summary, output_step, _ = sess.run(output,
                 feed_dict={x_placeholder: x, y_placeholder: y})
        
        summary_writer.add_summary(output_summary, output_step)
        
        #print it
        if epoch % 1000 == 0:
            plt.figure()
            plt.plot(output_x, output_y, 'o')
            plt.plot(output_x, output_y, '-')
            plt.show()
    
    summary_writer.close()
