#coding:utf-8
import tensorflow as tf
'''
正则化： 防止过拟合
'''

'''
1. tf.contrib.layers.apply_regularization: 
参数：
    * regularizer: 正则化函数，该函数输入为一个Tensor, 返回一个scalar tensor
    * weights_list: list of weights Tensors
返回：
    * A scalar, 所有weights的正则化penalty。
'''
w=tf.constant([10,20])
regular=tf.contrib.layers.apply_regularization(lambda x:tf.reduce_sum(x*x),[w])
sess=tf.Session()
out=sess.run(regular)
print(out)

'''
2. tf.contrib.layers.l1_regularizer:
返回计算l1正则项的函数
参数：
    * scale: A scalar
    * scope: 可选的scope name.
返回：
    * 计算l1正则的函数： l1(weights)
'''
l1_func=tf.contrib.layers.l1_regularizer(scale=0.1)
l1_reg=tf.contrib.layers.apply_regularization(l1_func,[w])
out=sess.run(l1_reg)
print(out)

'''
3. tf.contrib.layers.l2_regularizer:
'''

'''
4. tf.contrib.layers.sum_regularizer
'''