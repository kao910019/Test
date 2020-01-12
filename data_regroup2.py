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

encode_id = tf.constant([[0,   0],
                         [111, 0],
                         [111, 0],
                         [0,   0],
                         [0,   0],
                         [111, 111],
                         [111, 111],
                         [111, 111],
                         [111, 111],
                         [111, 111]])

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

hp_batch_size = 10
max_term_length = 25

sentence_ids = tf.concat([encode_id, decode_id], axis= 1)
batch_size, max_length = tf.shape(sentence_ids)[0], tf.shape(sentence_ids)[1]

encode_indice = tf.where(tf.not_equal(encode_id[:,0], 0))
decode_indice = tf.where(tf.not_equal(decode_id[:,0], 0))
encode_indice = tf.scatter_nd(encode_indice, tf.ones_like(encode_indice, tf.int32), [batch_size, 1])
decode_indice = tf.scatter_nd(decode_indice, tf.ones_like(decode_indice, tf.int32), [batch_size, 1])
decode_indice = tf.roll(decode_indice, shift=1, axis=0)

part_indice = tf.roll(tf.reshape(encode_indice * decode_indice, [-1]), shift=-1, axis=0)
part_indice = tf.scan(lambda a,x : a+x, part_indice, reverse = True)
part_indice = (tf.reduce_max(part_indice) - part_indice)

term_count = tf.reduce_max(tf.unique_with_counts(part_indice)[2])

sentence_part = tf.dynamic_partition(sentence_ids, part_indice, hp_batch_size)
sentence_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0), 
                    lambda: tf.concat([tf.reshape(part, [1, -1]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)], axis=1), 
                    lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)) for part in sentence_part], 0)

indice = tf.reshape(tf.where(tf.not_equal(sentence_regroup[:,0], 0)), [-1])
sentence_regroup = tf.gather(sentence_regroup, indice)

batch_size, max_length = tf.shape(sentence_regroup)[0], tf.shape(sentence_regroup)[1]

odd_indices = tf.matmul(tf.reshape(tf.range(1, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
even_indices = tf.matmul(tf.reshape(tf.range(0, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
partitions = tf.where(tf.not_equal(sentence_regroup, 0), even_indices, odd_indices)
sequence_parts = tf.dynamic_partition(sentence_regroup, partitions, max_term_length*4)
sentence_regroup = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(sentence_regroup))

with tf.Session() as sess:
    
    output = sess.run([encode_id, decode_id, sentence_regroup, encode_indice, part_indice])
    
    for out in output:
        print()
        print(out)

