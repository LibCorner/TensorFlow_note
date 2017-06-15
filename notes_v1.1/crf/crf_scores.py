#coding:utf-8
import tensorflow as tf

'''
1. tf.contrib.crf.crf_binary_score(tag_indices,sequence_lenths,transition_params)
计算biary scores of tag sequences.
参数：
    * tag_indices: shape=[batch_size,max_seq_len]
    * sequence_lengths: shape=[batch_size],序列长度
    * transition_params: shape=[num_tags,num_tags]
返回：
    binary_scores: shape=[batch_size]
'''
batch_size=16
max_seq_len=10

tag=tf.placeholder(dtype=tf.float32,shape=[batch_size,max_seq_len])
params=tf.Variable(tf.random_normal(shape=[4,4]))
scores=tf.contrib.crf.crf_binary_score(tag,sequence_lengths=tf.ones([batch_size])*10,transition_params=params
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a,b=sess.run([new_alphas,alphas],feed_dict={inputs:data})
    c=sess.run(params)
    print(c)
    print(a)
    print(b)
    print(a==b)
    print(a.shape)
    pprint(["可训练的参数：",tf.trainable_variables()]))