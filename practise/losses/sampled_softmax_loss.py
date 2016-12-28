#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

num_samples=10
target_vocab_size=1000
size=100


if num_samples>0 and num_samples<target_vocab_size:
    w=tf.Variable(tf.random_uniform([size,target_vocab_size],-1,1))
    #转置
    w_t=tf.transpose(w)
    b=tf.Variable(tf.zeros([target_vocab_size]))


inputs=tf.placeholder(tf.float64,shape=[None,size])
labels=tf.placeholder(tf.int64,shape=[None,1])
lables=tf.reshape(labels,[-1,1])
#参数类型转换成tf.float32
local_w_t=tf.cast(w_t,tf.float32)
local_b=tf.cast(b,tf.float32)
local_inputs=tf.cast(inputs,tf.float32)

#sampled_softmax_loss
loss=tf.nn.sampled_softmax_loss(local_w_t,local_b,local_inputs,labels,num_samples,target_vocab_size)
data=np.random.rand(10,100)
label_data=np.random.randint(low=0,high=target_vocab_size,size=(10,1))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print  sess.run(loss,feed_dict={inputs:data,labels:label_data})
