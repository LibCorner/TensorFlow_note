#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

batch_size=10
dim=30

def get_variable(in_dim,out_dim):
    w=tf.Variable(tf.random_uniform([in_dim,out_dim],-1.0,1.0))
    b=tf.Variable(tf.zeros(out_dim))
    return w,b
inputs=tf.placeholder(tf.float32,shape=[None,dim])
w1,b1=get_variable(dim,10)
out=tf.matmul(inputs,w1)+b1

w2,b2=get_variable(10,1)
out=tf.matmul(out,w2)+b2
#saver
#无参数，保存所有的变量,有参数保存指定变量
#saver=tf.train.Saver([w1,b1,w2,b2])
saver=tf.train.Saver()
data=np.ones([batch_size,dim])
with tf.Session() as sess:
    #加载
    #saver.restore(sess,"saver_wights.saver")
    #加载保存的变量，就不能再初始化变量,否则加载就无效
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    #保存
    saver.save(sess,"saver_wights1.saver")
    print(sess.run(out,feed_dict={inputs:data}))
    
