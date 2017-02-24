#coding:utf-8
import tensorflow as tf
import numpy as np


def prod(x):
    '''累乘，计算维度'''
    return tf.reduce_prod(x, reduction_indices=0, keep_dims=False)

def batch_flatten(x):
    '''
    把多维矩阵reshape成二维
    '''
    return tf.reshape(x,[-1,prod(tf.shape(x)[1:])])   

def flatten(x):
    '''
    把矩阵reshape成一维向量
    '''
    return tf.reshape(x, [-1])
    

input=tf.placeholder(dtype='float32',shape=[None,2,5])
#输出维度
print(input.get_shape())

#把输入reshape成[None,10]
input_1=batch_flatten(input)
print(input_1.get_shape())

W=tf.Variable(tf.random_uniform([10,5],-1.0,1.0))
out=tf.matmul(input_1,W)

print(tf.shape(out))

#获取Tensor变量的shape
print(out.get_shape())
print(W.get_shape())



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data=np.random.rand(100,2,5)
    res=sess.run(out,feed_dict={input:data})
    print(res.shape)
    
