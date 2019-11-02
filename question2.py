# -*- coding: utf-8 -*-
"""
    Question 2:
        
        There're two 2D - matrix called matrix1 and matrix2.
        Try to combine them and rebuild to another matrix like below.
        
        Notice: For increase calculation speed, try to not use for/while loop to get the goal.
                So you can use all tensorflow function except tf.while_loop()
        
        matrix1 = [[1,2,3,4,5,0,0,0,0,0],
                   [1,2,3,4,5,6,7,8,0,0],
                   [1,2,3,4,5,6,0,0,0,0],
                   [1,2,3,4,5,6,7,8,9,0],
                   [1,2,3,4,5,6,0,0,0,0]]
        shape = [5, 10]
        
        matrix2 = [[1,2,3,0,0,0,0,0,0,0],
                   [1,2,3,4,5,6,0,0,0,0],
                   [1,2,3,4,0,0,0,0,0,0],
                   [1,2,3,4,5,6,7,0,0,0],
                   [1,2,3,4,5,6,7,8,9,0]]
        shape = [5, 10]
        
        matrix3 = [[1,2,3,4,5,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0],
                   [1,2,3,4,5,6,7,8,1,2,3,4,5,6,0,0,0,0,0,0],
                   [1,2,3,4,5,6,1,2,3,4,0,0,0,0,0,0,0,0,0,0],
                   [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,0,0,0,0],
                   [1,2,3,4,5,6,1,2,3,4,5,6,7,8,9,0,0,0,0,0]]
        shape = [5, 20]
        
"""
import tensorflow as tf

matrix1 = tf.constant(
        [[1,2,3,4,5,0,0,0,0,0],
         [1,2,3,4,5,6,7,8,0,0],
         [1,2,3,4,5,6,0,0,0,0],
         [1,2,3,4,5,6,7,8,9,0],
         [1,2,3,4,5,6,0,0,0,0]])
    
matrix2 = tf.constant(
        [[1,2,3,0,0,0,0,0,0,0],
         [1,2,3,4,5,6,0,0,0,0],
         [1,2,3,4,0,0,0,0,0,0],
         [1,2,3,4,5,6,7,0,0,0],
         [1,2,3,4,5,6,7,8,9,0]])

# Rebuild input matrix.
matrix = tf.concat([matrix1,matrix2],axis=1)
shape = tf.shape(matrix)
# Indices use to divide vector.
num_partial = 10
odd_indices = tf.matmul(tf.reshape(tf.range(1, shape[0]*2, 2), [-1, 1]), tf.fill([1, shape[1]], 1))
even_indices = tf.matmul(tf.reshape(tf.range(0, shape[0]*2, 2), [-1, 1]), tf.fill([1, shape[1]], 1))
# Partial 0 and other number, then recombine it.
target = tf.where(tf.not_equal(matrix, 0), even_indices, odd_indices)
sequence_parts = tf.dynamic_partition(matrix, target, num_partial)
matrix3 = tf.reshape(tf.concat(sequence_parts, axis=0), shape)

with tf.Session() as sess:
    matrix1, matrix2, matrix3 = sess.run([matrix1, matrix2, matrix3])
    print("matrix1 =", matrix1)
    print("matrix2 =", matrix2)
    print("matrix3 =", matrix3)
    
    
