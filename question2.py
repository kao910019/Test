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

# --- Write your code below --- #