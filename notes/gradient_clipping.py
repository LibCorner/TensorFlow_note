#coding:utf-8

#Gradient Clipping
'''
Tensorflow提供了一些clipping函数可以添加到graph中。这些函数可以使用到一半的数据上，但是他们对处理梯度消失和梯度爆炸尤其有用。

1. tf.clip_by_value(t,clip_value_min,clip_value_max,name=None):
    把tensor值限制在特定的min和max之间。
    
    给定一个Tensor t, 这个操作返回一个和t具有相同type和shape的tensor,它的值在clip_value_min和clip_value_max之间。
    小于clip_value_min的值设置为clip_value_min,大于clip_value_max的值设置为clip_value_max。

2. tf.clip_by_norm(t,clip_norm,axes=None,name=None):
    
    Clips tensor value to a maximum L2-norm，限制tensor的最大L2-norm。
    
    如果t的L2-norm小于或等于clip_norm, t不修改。如果L2-norm大于clip_norm, t的值修改为：t*clip_norm/l2norm(t).
    
    如果t是一个矩阵并且axes==[1],每一行的L2-norm要小于clip_norm,如果axes==[0] 每一列的L2-norm要小于clip_norm.

3. tf.clip_by_average_norm(t,clip_norm,name=None):
    限制tensor的最大平均L2-norm

4.  tf.clip_by_global_norm(t_list,clip_norm,use_norm=None,name=None)   
    
'''

import tensorflow as tf

a=tf.Variable(tf.truncated_normal([10,5],stddev=5))
b=tf.clip_by_value(a,-1.0,1.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o1,o2=sess.run([a,b])
    print(o1)
    print(o2)