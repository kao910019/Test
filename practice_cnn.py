# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from tensorflow.examples.tutorials.mnist import input_data
# hyper parameters
num_iters = 10000
learning_rate = 8e-4
batch_size = 50

# Load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#placeholder
x_placeholder = tf.placeholder(tf.float32, [None,None])
y_placeholder = tf.placeholder(tf.float32, [None,None])

def Network(x, y):
    with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
        # change shape to [batch_size, 1]
        bx = tf.reshape(x, [-1, 28, 28, 1])
        by = tf.argmax(y, axis=-1)
        
        #Conv & Pooling
        conv1 = tf.layers.conv2d(bx, 64, 3, strides=(5, 5), padding="same")
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size = 2, strides = (2, 2), padding="same")
        conv2 = tf.layers.conv2d(pool1, 128, 3, strides=(5, 5), padding="same")
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size = 2, strides = (2, 2), padding="same")
        
        #Classifier
        output = tf.layers.dense(pool2, 512, activation=tf.nn.relu)
        output = tf.nn.dropout(output, 0.9)
        output = tf.layers.dense(output, 10)
        output = tf.squeeze(output)
        predict_y = tf.nn.softmax(output, axis=-1)
        
        #Cross entropy
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=by, logits=predict_y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict_y, axis=-1), by), tf.float32))
        
        #minimize
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
    return bx, by, predict_y, loss, accuracy, train_op

with tf.Session() as sess:
    with sess.graph.as_default():
        # Draw your neural network here.
        output = Network(x_placeholder, y_placeholder)
    
    #init variables
    sess.run(tf.global_variables_initializer())
    
    #training 10000 times
    for epoch in range(num_iters):
        #Get data
        batch = mnist.train.next_batch(batch_size)
        
        x, y, pred_y, loss, accuracy, _ = sess.run(output, feed_dict={x_placeholder: batch[0], y_placeholder: batch[1]})
        
        # print it
        if epoch % 1000 == 0:
            index = np.random.randint(0, batch_size)
            
            x = np.reshape(x, [-1, 28, 28])
            plt.imshow(x[index])
            plt.show(block=False)
            plt.pause(0.01)
            
            prop = pred_y[index]
            pred_y = np.argmax(prop)
            print("Lable = {}, Predict = {} : {:3.1f}%".format(y[index], pred_y, prop[pred_y]*100.0))
            print("Loss = {:.3f}  Accuracy = {:.3f}".format(loss, accuracy))
            
tf.reset_default_graph()
