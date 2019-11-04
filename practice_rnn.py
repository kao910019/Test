# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# hyper parameters
num_iters = 10000
learning_rate = 8e-4
batch_size = 50

# Load stock price
# 中國鋼鐵 股票
steel_price = pd.read_csv('2002.TW.csv')
# 淡水河谷 股票
iron_price = pd.read_csv('VALE.csv')

# Combine the price with date
stock_price = pd.merge(steel_price, iron_price, on='Date', how='outer')
steel_price = stock_price['Open_x'].to_numpy()
iron_price = stock_price['Open_y'].to_numpy()
# remove nan price
sp_list, ip_list = [], []
for sp, ip in zip(steel_price, iron_price):
    if not np.isnan(sp) and not np.isnan(ip):
        sp_list.append(sp)
        ip_list.append(ip)
steel_price = np.array(sp_list)
iron_price = np.array(ip_list)

plt.plot(steel_price)
plt.plot(iron_price)
plt.show()

# Create label what we want to predict.
label = []
for index in range(steel_price.shape[0] - 1):
    if steel_price[index] < steel_price[index+1]:
        label.append(0)
    else:
        label.append(1)
label = np.array(label, np.int32)
        
#placeholder
x_placeholder = tf.placeholder(tf.float32, [None])
x2_placeholder = tf.placeholder(tf.float32, [None])
# Predict up and down
y_placeholder = tf.placeholder(tf.int64, [None])
# Predict price
#y_placeholder = tf.placeholder(tf.float32, [None])
   
def Network(x, x2, y):
    with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
        # change shape to [batch_size, 1]
        bx = tf.reshape(x, [1, -1, 1])
        bx2 = tf.reshape(x2, [1, -1, 1])
        
        bx = tf.concat([bx, bx2], axis=-1)
        
        # Predict up and down
        by = tf.reshape(y, [1, -1])
        # Predict price
#        by = tf.reshape(y, [1, -1, 1])
        
        # create RNN
        cell_list = []
        for i in range(2):
            cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 0.9)
            cell_list.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        
        output, states = tf.nn.dynamic_rnn(
                cell, bx, initial_state=cell.zero_state(1, dtype=tf.float32), dtype=tf.float32)
        
#        # Predict up and down
        output = tf.layers.dense(output, 128, activation=tf.nn.relu)
        output = tf.nn.dropout(output, 0.9)
        output = tf.layers.dense(output, 2)
        predict_y = tf.nn.softmax(output, axis=-1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=by, logits=predict_y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict_y, axis=-1), by), tf.float32))
        
        # Predict price
#        output = tf.layers.dense(output, 128, activation=tf.nn.relu)
#        predict_y = tf.layers.dense(output, 1)
#        loss = tf.losses.mean_squared_error(by, predict_y)
        
        #minimize
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
    # Predict up and down
    return bx, by, predict_y, loss, accuracy, train_op
    # Predict price
#    return bx, by, predict_y, loss, train_op

with tf.Session() as sess:
    with sess.graph.as_default():
        # Draw your neural network here.
        output = Network(x_placeholder, x2_placeholder, y_placeholder)
    
    #init variables
    sess.run(tf.global_variables_initializer())
    
    process_bar = tqdm(range(num_iters))
    #training 10000 times
    for epoch in process_bar:
        # Predict up and down
        bx, by, predict_y, loss, accuracy, _ = sess.run(output, feed_dict={x_placeholder: steel_price[:-1],
                                                                           x2_placeholder: iron_price[:-1],
                                                                           y_placeholder: label})
        process_bar.set_description("Loss = {:.3f}, Acc = {:.3f}".format(loss, accuracy))
        
        # Predict price
#        bx, by, predict_y, loss, _ = sess.run(output, feed_dict={x_placeholder: steel_price[:-1],
#                                                                 x2_placeholder: iron_price[:-1],
#                                                                 y_placeholder: steel_price[1:]})
#        if epoch % 100 == 0:
#            plt.plot(np.reshape(by, [-1]))
#            plt.plot(np.reshape(predict_y, [-1]))
#            plt.show()
            
tf.reset_default_graph()
