# -*- coding: utf-8 -*-
#GradientDscent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

samples_num=500

x=np.random.rand(samples_num)
y=x*3+0.5+np.random.normal(0,0.5,size=samples_num)

plt.scatter(x,y,s=20)
plt.show()

inputs=tf.placeholder(tf.float32,shape=[samples_num])
labels=tf.placeholder(tf.float32,shape=[samples_num])
w=tf.Variable(tf.truncated_normal(shape=[1],stddev=0.1),name='w')
b=tf.Variable(tf.zeros([1]),name='b')
out=tf.mul(inputs,w)+b
#计算损失
loss=tf.reduce_mean(tf.square(out-labels))
#GradientDscentOptimizer
#opt=tf.train.GradientDescentOptimizer(0.3)

#AdadeltaOptimizer
#opt=tf.train.AdadeltaOptimizer(1.0,rho=0.95,epsilon=1e-3)

#AdamOptimizer
#opt=tf.train.AdamOptimizer(learning_rate=0.1,beta1=0.9,beta2=0.99,epsilon=1e-8)

#AdagradOptimizer
opt=tf.train.AdagradOptimizer(learning_rate=0.5,initial_accumulator_value=0.1)
#train_op=opt.minimize(loss)

#计算梯度
grad_vars=opt.compute_gradients(loss,var_list=[w,b])
#对梯度进行约束
caped_grads=[(tf.clip_by_value(g,-1.0,1.0),v) for g,v in grad_vars]
#应用梯度
train_op=opt.apply_gradients(grad_vars)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    feed_dict={inputs:x,labels:y}
    for i in range(100):
        #print(sess.run(grad_vars,feed_dict))
        _,y_pre,err=sess.run([train_op,out,loss],feed_dict=feed_dict)
        if i%20==0:
            print(err)
            pre_w,pre_b=sess.run([w,b])
            print(pre_w,pre_b)
            plt.scatter(x,y)
            plt.plot(x,y_pre,c='red')
            plt.show()
    
    
