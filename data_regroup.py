# Combine encode_ids and decode_ids and regroup them to one-to-one form.
# Use in Multiterm Dialogue that user or robot may listen and none reply in conversation.
# 
# Encode_ids
#
# [[  0   0]
#  [111   0]
#  [111   0]
#  [  0   0]
#  [  0   0]
#  [111 111]
#  [111 111]
#  [111 111]
#  [111 111]
#  [111 111]]
#
# Decode_ids
#
# [[222 222]
#  [222   0]
#  [222 222]
#  [222   0]
#  [222   0]
#  [  0   0]
#  [  0   0]
#  [222 222]
#  [222 222]
#  [  0   0]]
# 
# Output_ids
#
# [[111 222   0   0   0   0   0   0   0   0   0   0]
#  [111 222 222 222 222   0   0   0   0   0   0   0]
#  [111 111 111 111 111 111 222 222   0   0   0   0]
#  [111 111 222 222   0   0   0   0   0   0   0   0]
#  [111 111   0   0   0   0   0   0   0   0   0   0]]

import numpy as np
import tensorflow as tf

encode_id = tf.constant([[0,   0, 0],
                         [111, 0, 0],
                         [111, 0, 0],
                         [0,   0, 0],
                         [0,   0, 0],
                         [111, 111, 0],
                         [111, 111, 0],
                         [111, 111, 0],
                         [111, 111, 0],
                         [111, 111, 0]])

decode_id = tf.constant([[222, 222],
                         [222, 0],
                         [222, 222],
                         [222, 0],
                         [222, 0],
                         [0,   0],
                         [0,   0],
                         [222, 222],
                         [222, 222],
                         [0,   0]])

target_logits = tf.constant([[[1.0,2.0], [1.0,2.0]],
                             [[2.0,2.0], [1.0,2.0]],
                             [[3.0,2.0], [1.0,2.0]],
                             [[4.0,2.0], [1.0,2.0]],
                             [[5.0,2.0], [1.0,2.0]],
                             [[6.0,2.0], [1.0,2.0]],
                             [[7.0,2.0], [1.0,2.0]],
                             [[8.0,2.0], [1.0,2.0]],
                             [[9.0,2.0], [1.0,2.0]],
                             [[10.0,2.0], [1.0,2.0]]])


hp_batch_size = 10
max_term_length = 25

def resort_zero(inputs, max_batch_size):
    """  
    Inputs 2D tensor that can resort zero in tensor to the end.
    
    Parameters
    ----------
    inputs : Tensor. shape = [batch_size, max_length]
    max_batch_size : Int32 or Int64 (can't input a tensor)
    
    Returns
    -------
    outputs : Tensor. shape = [batch_size, max_length]
    """
    with tf.name_scope("resort_zero"):
        batch_size, max_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        odd_indices = tf.matmul(tf.reshape(tf.range(1, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
        even_indices = tf.matmul(tf.reshape(tf.range(0, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
        partitions = tf.where(tf.not_equal(inputs, 0), even_indices, odd_indices)
        sequence_parts = tf.dynamic_partition(inputs, partitions, max_batch_size*4)
        outputs = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(inputs))
    return outputs, partitions
    
def pad_complement_each_other(inputs1, inputs2, pad=None, axis=None):
    max_axis_shape = tf.reduce_max([tf.shape(inputs1)[axis], tf.shape(inputs2)[axis]])
    inputs1 = tf.concat([inputs1, tf.fill([max_axis_shape - tf.shape(inputs1)[i] if i==axis else tf.shape(inputs1)[i] for i in range(tf.shape(inputs1).shape[0])], pad)], axis=axis)
    inputs2 = tf.concat([inputs2, tf.fill([max_axis_shape - tf.shape(inputs2)[i] if i==axis else tf.shape(inputs2)[i] for i in range(tf.shape(inputs2).shape[0])], pad)], axis=axis)
    return inputs1, inputs2, max_axis_shape

def summary_sentence_regroup(source_id, target_id, pad_id, source_tag='Source', target_tag='Target'):
    batch_size = tf.shape(source_id)[0]
    
    source_id, target_id, max_time = pad_complement_each_other(source_id, target_id, pad_id, axis=1)
    sentence_ids = tf.reshape(tf.concat([source_id, target_id], axis= 1), [-1, max_time])
    zero_indice = tf.where(tf.not_equal(sentence_ids[:, 0], 0))
    
    tag_labels = tf.reshape(tf.concat([tf.fill([batch_size, 1], 0), tf.fill([batch_size, 1], 1)], axis=1), [-1, 1])
    
    sentence_ids = tf.gather_nd(sentence_ids, zero_indice)
    tag_labels = tf.gather_nd(tag_labels, zero_indice)
    
    label_batch_size = tf.shape(tag_labels)[0]
    tag_labels = tf.where(tf.equal(tag_labels, 0), tf.fill([label_batch_size, 1], source_tag), tf.fill([label_batch_size, 1], target_tag))
    return tag_labels, sentence_ids

def regroup_sentence_one2one_form(source_id, target_id, target_logits=None):
    # sentence_regroup
    tag, sentence_id = summary_sentence_regroup(source_id, target_id, 0, source_tag=False, target_tag=True)
    max_length = tf.shape(sentence_id)[1]
    # sentence_slice
    front_slice_indice = tf.where(tf.equal(tag, False))[0,0]
    back_slice_indice = tf.where(tf.equal(tag, True))[-1,0] + 1
    sentence_id = sentence_id[front_slice_indice:back_slice_indice]
    tag = tag[front_slice_indice:back_slice_indice]
    
    if target_logits != None:
        zero_indice = tf.where(tf.not_equal(target_id[:, 0], 0))
        target_logits = tf.gather(target_logits, zero_indice)[front_slice_indice:,0]
        target_logits = tf.concat([target_logits, tf.zeros([tf.shape(target_logits)[0], max_length - tf.shape(target_logits)[1], 2], tf.float32)], axis=1)
        logits_labels = tf.gather(target_id, zero_indice)[front_slice_indice:,0]
        logits_labels = tf.concat([logits_labels, tf.zeros([tf.shape(logits_labels)[0], max_length - tf.shape(logits_labels)[1]], tf.int32)], axis=1)

    # part indice
    tag = tf.cast(tf.logical_xor(tag, tf.roll(tag, shift=-1, axis=0)), tf.int32)
    part_indice = tf.scan(lambda a,x : a+x, tag, reverse = True)
    part_indice = tf.reshape(tf.reduce_max(part_indice) - part_indice, [-1])
    term_count = tf.reduce_max(tf.unique_with_counts(part_indice)[2])
    # regroup part sentence
    sentence_part = tf.dynamic_partition(sentence_id, part_indice, hp_batch_size)
    sentence_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0), 
                        lambda: tf.concat([tf.reshape(part, [1, -1]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)], axis=1), 
                        lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)) for part in sentence_part], 0)
    
    if target_logits != None:
        logit_indice = tf.reshape(tf.gather(part_indice, tf.where(tf.equal(part_indice % 2, 1))), [-1])
        logits_part = tf.dynamic_partition(target_logits, logit_indice, hp_batch_size)
        labels_part = tf.dynamic_partition(logits_labels, logit_indice, hp_batch_size)
        new_logits_part, new_labels_part = [], []
        for i, (array1, array2) in enumerate(zip(logits_part, labels_part)):
            if i % 2 != 0:
                new_logits_part.append(array1)
                new_labels_part.append(array2)
        logits_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0),
                        lambda: tf.concat([tf.reshape(part, [1, -1, 2]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length, 2], tf.float32)], axis=1), 
                        lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length, 2], tf.float32)) for part in new_logits_part], 0)
        labels_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0),
                        lambda: tf.concat([tf.reshape(part, [1, -1]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)], axis=1), 
                        lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)) for part in new_labels_part], 0)
        labels_regroup, partitions = resort_zero(labels_regroup, hp_batch_size)
        sequence_parts = tf.dynamic_partition(logits_regroup, partitions, hp_batch_size*4)
        target_logits = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(logits_regroup))
        indice = tf.reshape(tf.where(tf.not_equal(labels_regroup[:,0], 0)), [-1])
        target_logits = tf.gather(target_logits, indice)
        
    # sort zero to behind
    sentence_regroup, _ = resort_zero(sentence_regroup, hp_batch_size)
    # remove pad sentence
    indice = tf.reshape(tf.where(tf.not_equal(sentence_regroup[:,0], 0)), [-1])
    sentence_regroup = tf.gather(sentence_regroup, indice)
    # split two part
    max_length = tf.shape(sentence_regroup)[1]
    sentence_regroup = tf.reshape(sentence_regroup, [-1, max_length*2])
    source_id = sentence_regroup[:,:max_length]
    target_id = sentence_regroup[:,max_length:]
    return source_id, target_id, target_logits
      
encode_id, decode_id, target_logits = regroup_sentence_one2one_form(encode_id, decode_id, target_logits)

with tf.Session() as sess:
    
    output = sess.run([encode_id, decode_id, target_logits])
    
    for out in output:
        print()
        print(out)
