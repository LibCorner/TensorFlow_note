#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


#conv1d
x1=tf.placeholder(tf.float32,shape=[None,10,100])
#Filter
W=tf.Variable(tf.random_uniform([3,100,50],-1.0,1.0))
y1=tf.nn.conv1d(x1,W,1,"SAME")
#扩展成4维[batch,width,channel]-->[batch,1,width,channel]
y1=tf.expand_dims(input=y1,dim=1)
#max pooling
max1=tf.nn.max_pool(y1,[1,1,5,1],[1,1,1,1],"VALID")


#conv2d
x2=tf.placeholder(tf.float32,shape=[None,100,100,1])
W2=tf.Variable(tf.random_uniform([5,5,1,3],-1.0,1.0))
y2=tf.nn.conv2d(x2,W2,[1,1,1,1],"SAME")
max2=tf.nn.max_pool(y2,[1,5,5,1],[1,1,1,1],"VALID")

sess=tf.Session()
sess.run(tf.initialize_all_variables())
data=np.random.rand(5,10,100)
print sess.run(y1,feed_dict={x1:data})
print sess.run(max1,feed_dict={x1:data})

data2=np.random.rand(5,100,100,1)
print sess.run(y2,feed_dict={x2:data2})
print sess.run(max2,feed_dict={x2:data2})
