# -*- coding: utf-8 -*-

#Optimizer
'''
class tf.train.Optimizer: 所有Optimizer的基类，提供了对loss计算梯度和把梯度应用到variables上的方法。
不能直接使用该类的API,而是通过其子类的实例来调用。

Usage:
    #创建Optimizer
    opt=tf.train.GradientDescentOptimizer(learning_rate=0.1)
    #通过更新一个variables列表来最小化cost的操作。
    #cost是一个Tensor, list of variables是tf.Variable对象列表。
    opt_op=opt.minize(cost,var_list=<list of variables>)
    #运行opt_op来训练
    opt_op.run()
'''

#在应用梯度前对梯度进行处理
'''    
调用minimize()方法会计算梯度并应用到变量上，
如果要在应用梯度之前对梯度进行处理通过一下三步使用Optimizer:
    1. 使用compute_gradients()计算梯度
    2. 对梯度进行处理
    3. 使用apply_gradients()应用梯度
    
    
    #创建Optimizer
    opt=tf.train.GradientDescentOptimizer(learning_rate=0.1)
    
    #计算梯度
    grads_and_vars=opt.compute_gradients(loss,<list of variables>)
    
    #grads_and_vars是a list of tuples (gradient,variable).
    #处理梯度MyCapper(gradient)
    capped_grads_and_vars=[(MyCapper(gv[0]),gv[1]) for gv in grads_and_vars]
    
    #使用apply_gradients应用capped gradients
    opt.apply_gradients(capped_grads_and_vars)
'''

#类方法
'''
1. tf.train.Optimizer.__init__(use_locking,name):
    构造方法。
    参数：
        use_locking: Bool.如果True使用locks防止对变量同步更新。
        name:The name to use for accumulators create for the optimizer.
        
2. tf.train.Optimizer.minimize(loss,
                               global_step=None,
                               var_list=None,
                               gate_gradients=1,
                               aggregation_method=None,
                               colocate_gradients_with_ops=False,
                               name=None,
                               grad_loss=None):
    通过更新var_list来最小化loss，这个方法简单的把compute_gradients()和apply_gradients()结合起来。
    如果要对梯度进行处理可以显式的调用conpute_gradients()和apply_gradients()方法。
    
    参数：
        loss: A Tensor containing the value to minize.
        global_step: 可选Variable, 每次更新变量加1.
        var_list:Optional list of Variable objects to update to minimize loss.
                Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
        gate_gradients:How to gate the computation of gradients.可以为GATE_NONE,GATE_OP,或GATE_GRAPH.
        aggregation_method:
    返回：
        更新var_list的Operation. 如果global_step不是None,这个operation会对global_step加1.

3. tf.train.Optimizer.compute_gradients(loss,
                                        var_list=None,
                                        gate_gradients=1,
                                        aggregation_method=None,
                                        colocate_gradients_with_ops=False,
                                        grad_loss=None):
    计算loss对var_list中的变量的梯度，返回a list of (gradient,variable) pairs.

4. tf.train.Optimizer.apply_gradients(grads_and_vars,
                                      global_step=None,
                                      name=None):
    把梯度应用到变量，返回应用梯度更新变量的Operation.
'''


#GradientDescentOptimizer
'''
class tf.train.GradientDescentOptimizer:
    def __init__(self,learning_rate,use_locking=False,name='GradientDescent')
参数：
    learning_rate: A Tensor或者浮点数。学习速率。
    use_locking: 如果True就对更新操作使用locks.
    name: 操作的名字。
'''


#AdadeltaOptimizer
'''
实现Adadelta算法的Optimizer,see:`https://arxiv.org/pdf/1212.5701v1.pdf`.
class tf.train.AdadeltaOptimizer:
    def __init__(self,
                 learning_rate=0.001,
                 rho=0.95,  
                 epsilon=1e-08,
                 use_locking=False
                 ,name='Adadelta') 
参数：
    learning_rate: 学习速率，Tensor或float
    rho: The decay rate, Tensor或float
    epsilon: A constant epsilon used to better conditioning the grad update.
    use_locking: 如果True就对更新操作加锁。
    name:
'''


#AdagradOptimizer
'''
实现了Adagrad算法的Optimizer,see:`http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf`
class tf.train.AdagradOptimizer:
    def __init__(self,
                 learning_rate,
                 initial_accumulator_value=0.1,
                 use_locking=False,
                 name='Adagrad')
    
参数：
   initial_accumulator_value: accumulators的初始值。
'''

#MomentumOptimizer
'''
实现了Momentum算法的Optimizer.
class tf.train.MomentumOptimizer:
    def __init__(self,
                 momentum,
                 use_locking=False,
                 name='Momentum',
                 use_nesterov=False)
'''

#AdamOptimizer
'''
实现了Adam算法的Optimizer,see `https://arxiv.org/pdf/1412.6980.pdf`.
class tf.train.AdamOptimizer(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 use_locking=False,
                 name='Adam')
参数：
    learning_rate:学习速率。
    beta1: A float value or a constant float tensor. 
            The exponential decay rate for the 1st moment estimates.
    beta2:The exponential decay rate for the 2nd moment estimates.
    epsilon: A small constant for numerical stability.

'''

#RMSPropOptimzer
'''
class tf.trian.RMSPropOptimizer:
    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 use_locking=False,
                 centered=False,
                 name='RMSProp')
'''


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
    