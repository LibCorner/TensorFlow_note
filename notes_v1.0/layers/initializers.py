#coding:utf-8
import tensorflow as tf
'''
变量的初始化
'''

'''
1. tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
2. tf.contrib.layers.xavier_initializer_conv2d(uniform=True,seed=None,dtype=tf.float32)
返回使用`Xavier`初始化得到权重的初始值。这个initialzer可以保持所有层的梯度基本一致。
参数：
    * uniform: Ture,使用均匀分布, False:使用正态分布。
        (1) 使用均匀分布时,初始值的范围为[-x,x], x=sqrt(6./(in+out))
        (2) 使用正态分布时，初始值的deviation为：sqrt(3./(in+out))
参考论文:
    {Xavier Glorot and Yoshua Bengio (2010): 
    Understanding the difficulty of training deep feedforward neural networks. 
    International conference on artificial intelligence and statistics.}

'''

'''
3. tf.contrib.layers.variance_scaling_initializer
'''