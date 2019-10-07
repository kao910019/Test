# -*- coding: utf-8 -*-
"""
   Code from : https://stackoverflow.com/questions/42858785/connect-input-and-output-tensors-of-two-different-graphs-tensorflow
"""
import tensorflow as tf
from tensorflow.python.framework import meta_graph

with tf.Graph().as_default() as graph1:
    inputs1 = tf.placeholder(tf.float32, (None, 2), name='input1')
    v1 = tf.Variable(1, name='v1')
    output1 = tf.identity(inputs1, name='output1')
    print("G1:",tf.global_variables())

with tf.Graph().as_default() as graph2:
    inputs2 = tf.placeholder(tf.float32, (None, 2), name='input2')
    v2 = tf.Variable(1, name='v2')
    output2 = tf.identity(inputs2, name='output2')
    print("G2:",tf.global_variables())


graph = tf.get_default_graph()
x = tf.placeholder(tf.float32, (None, 2), name='input')


meta_graph1 = tf.train.export_meta_graph(graph=graph1)
meta_graph.import_scoped_meta_graph(meta_graph1, input_map={'input1': x}, import_scope='graph1')
out1 = graph.get_tensor_by_name('graph1/output1:0')

meta_graph2 = tf.train.export_meta_graph(graph=graph2)
meta_graph.import_scoped_meta_graph(meta_graph2, input_map={'input2': out1}, import_scope='graph2')
out2 = graph.get_tensor_by_name('graph2/output2:0')

print(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    tf.summary.FileWriter("tensorboard", sess.graph)
    
    outputs = sess.run(x, feed_dict={x: [[8,7]]})
    print(outputs)
    outputs = sess.run(out1, feed_dict={x: [[6,7]]})
    print(outputs)
    outputs = sess.run(out2, feed_dict={x: [[5,7]]})
    print(outputs)

