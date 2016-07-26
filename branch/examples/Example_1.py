# -*- coding: utf-8 -*-
#HelloWorld

import tensorflow as tf
import numpy as np

#随机数据
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.1,0.2],x_data)+0.3

#定义tensorflow变量
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))
#计算过程
y=tf.matmul(w,x_data)+b

#计算cost
loss=tf.reduce_mean(tf.square(y-y_data))
#定义优化器
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化变量
init=tf.initialize_all_variables()

#启动图
sess=tf.Session()
sess.run(init)
#拟合平面
for step in xrange(0,201):
    sess.run(train)
    if step%20==0:
        print step, sess.run(w),sess.run(b)
        
