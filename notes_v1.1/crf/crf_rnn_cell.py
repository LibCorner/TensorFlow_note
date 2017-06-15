#coding:utf-8
import tensorflow as tf
import numpy as np
from pprint import pprint

'''
tf.contrib.crf.CrfForwardRnnCell: 计算linear-chain CRF的alpha值
属性：
    * ouput_size
    * state_size
方法：
    (1) __init__(transition_params):初始化
        * transition_params: shape为[num_tags,num_tags]的binary potential矩阵。
    (2)__call__(inputs,state,scope=None)：
    参数：
        * inputs: shape=[batch_size,num_tags]
        * state: shape=[batch_size,num_tags], 包含previous alpha values
        * scope: unsed variable scope of thie cell.
    返回：
        * new_alphas,new alphas: shape为[batch_size,num_tags],值为新的alpha值。
    (3)zero_state(batch_size,dtype)
'''
batch_size=16
data=np.random.rand(batch_size,4)
tf.reset_default_graph()

inputs=tf.placeholder(dtype=tf.float32,shape=[batch_size,4])
params=tf.Variable(tf.random_normal(shape=[4,4]))
crf_cell=tf.contrib.crf.CrfForwardRnnCell(transition_params=params)
with tf.variable_scope("crf") as scope:
    init_state=crf_cell.zero_state(batch_size,dtype=tf.float32)
    prob=tf.nn.softmax(inputs,dim=-1)
    new_alphas,alphas=crf_cell(prob,state=init_state)
    pprint(new_alphas)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a,b=sess.run([new_alphas,alphas],feed_dict={inputs:data})
    c=sess.run(params)
    print(c)
    print(a)
    print(b)
    print(a==b)
    print(a.shape)
    pprint(["可训练的参数：",tf.trainable_variables()])