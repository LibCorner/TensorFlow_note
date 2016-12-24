#-*- coding:utf-8-*
import tensorflow as tf
import numpy as np

is_training=True
keep_prob=0.8

x=tf.placeholder(tf.float64,shape=[None,10])
if is_training and keep_prob<1:
    out=tf.nn.dropout(x,keep_prob)
data=np.ones((5,10))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(out,feed_dict={x:data})
