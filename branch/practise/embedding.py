#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

ids=np.random.randint(5,10)
emb=np.random.rand(1000,20)

x=tf.placeholder(tf.int32,shape=[None,10])
embedding=tf.nn.embedding_lookup(params=emb,ids=x)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(embedding,feed_dict={x:ids})
