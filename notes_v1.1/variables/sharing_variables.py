#coding:utf-8
import tensorflow as tf

'''
1. tf.get_variable
'''
tf.reset_default_graph()
with tf.variable_scope("foo") as scope:
    w=tf.get_variable("w",initializer=[1,2,3])
    v=tf.get_variable("v",initializer=[4,5,6])
    print(w.name)
    print(v.name)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out=sess.run(w)
    print(out)
    
tf.contrib.legacy_seq2seq.model_with_buckets