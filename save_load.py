# -*- coding: utf-8 -*-
import os
import tensorflow as tf

def variable_loader(session, result_dir, var_list = tf.global_variables(), max_to_keep=5):
    ckpt = tf.train.get_checkpoint_state(result_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("# Find checkpoint file:", ckpt.model_checkpoint_path)
        saver = tf.train.Saver(var_list, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1.0)
        print("# Restoring model weights ...")
        saver.restore(session, ckpt.model_checkpoint_path)
        return saver, True
    saver = tf.train.Saver(var_list, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1.0)
    return saver, False

SYSTEM_ROOT = os.path.abspath(os.path.dirname(__file__))
RESULT_DIR = os.path.join(SYSTEM_ROOT, 'Result')
RESULT_FILE = os.path.join(RESULT_DIR, 'save')

with tf.Session() as sess:
    
    graph = tf.get_default_graph()
    with graph.as_default():
        var = tf.Variable(0)
        
    saver, _ = variable_loader(sess, RESULT_DIR)
    
    print(sess.run(tf.assign_add(var, 1)))
    
    saver.save(sess, RESULT_FILE, global_step = 1)
