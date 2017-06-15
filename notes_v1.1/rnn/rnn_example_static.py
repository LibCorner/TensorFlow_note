#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs

batch_size=16
length=10
unit_num=20
'''
使用tf.reset_default_graph()或with来重置图，避免一些因为变量名字冲突而出现的问题
'''
#重置图
tf.reset_default_graph()

data=np.random.randint(low=0,high=1000,size=(batch_size,length))

idx=tf.placeholder(dtype=tf.int32,shape=[None,None])
embedding=np.random.rand(1000,50)
#embbedding
inputs=tf.nn.embedding_lookup(embedding,idx)
    
'''使用tf.contrib.rnn.static_rnn'''
#unpack成sequence
inputs=tf.transpose(inputs,perm=[1,0,2])
inputs=tf.unstack(inputs,num=length)
with tf.variable_scope("static_rnn") as scope:
    cell=tf.contrib.rnn.BasicLSTMCell(unit_num,forget_bias=1.0,state_is_tuple=True,activation=tf.tanh)
    #rnn
    output,states=tf.contrib.rnn.static_rnn(cell,inputs,dtype=tf.float64)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs,states=sess.run([output,states],feed_dict={idx:data})
    print(outputs)