# -*- coding: utf-8 -*-
# Open your cmd and type:
# Tensorboard --logdir="<your data place>" --port:<default: 6006>
# And connect to 127.0.0.1:6006
import tensorflow as tf

with tf.Session() as sess:
    with sess.graph.as_default():
        # Draw your neural network here.
        
        test = tf.placeholder(tf.float32, [None], name="test")
        
    tf.summary.FileWriter("tensorboard", sess.graph)
