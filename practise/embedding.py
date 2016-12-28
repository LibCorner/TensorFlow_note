#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

ids=np.random.randint(low=0,high=1000,size=(10,10))
emb=np.random.rand(1000,20)

x=tf.placeholder(tf.int32,shape=[None,10])
#embedding不作为训练参数
embedding=tf.nn.embedding_lookup(params=emb,ids=x)

#embedding作为训练参数
init_emb=tf.Variable(emb)
embeded=tf.nn.embedding_lookup(init_emb,ids=x)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run([embedding,embeded],feed_dict={x:ids})

