# -*- coding: utf-8 -*-
import os
import tensorflow as tf

def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist, var_value = [], []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = []
    for i, tensor_name in enumerate(loaded_tensors[0]):
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            full_var_list.append(tensor_aux)
        except:
            print('* Not found: '+tensor_name)
    return full_var_list

def variable_loader(session, result_dir, var_list = tf.global_variables(), max_to_keep=5):
    ckpt = tf.train.get_checkpoint_state(result_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("# Find checkpoint file:", ckpt.model_checkpoint_path)
        restored_vars  = get_tensors_in_checkpoint_file(file_name = ckpt.model_checkpoint_path)
        tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
        saver = tf.train.Saver(tensors_to_load, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1.0)
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
