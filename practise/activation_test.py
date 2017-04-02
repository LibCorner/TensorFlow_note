# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

'''
一、拟合函数
激活函数测试，在一个深度网络中，使用不同的激活函数拟合一个函数（平方，三次方等函数）
1. 数据和标签必须规则化到（-1,1）之间或者(0,1)之间， 一是方便在最后使用非线性函数(tanh,sigmoid),二是方便训练，是模型能更好的拟合，否则初始化损失可能太大，容易导致欠拟合
2. 不使用激活函数，只能拟合大概的趋势,直线拟合，不能拟合曲线
3. 只是用relu激活函数，也只能线性拟合，无法拟合曲线
4. relu函数的优点是在深度网络中能够解决梯度弥散的问题，具有稀疏激活的特性。relu函数主要用特征提取，来抽取重要的特征比较好。
5. 前面使用relu激活函数，最后一层使用tanh激活，效果也不太好
6. 前面几层使用relu激活函数，倒数第二层使用tanh或sigmoid，最后一层不使用激活函数效果最好，使用tanh,sigmoid效果也不错
7. 使用sigmoid或tanh激活函数，最后一层不使用激活函数，效果也很好
'''
'''
二、XOR分类问题
1. 最后一层用sigmoid分类
2. 前面全用tanh效果很差，前面全用relu效果一般, 全部用sigmoid效果好
3. 2个tanh加1个relu结合使用效果好很多
'''



def activation(x):
    return tf.nn.relu(x)
def inference(x,input_dim=2):
    
    w=tf.Variable(tf.random_normal(shape=[input_dim,100],mean=0.,stddev=0.1))
    
    w1=tf.Variable(tf.random_normal(shape=[100,100],mean=0.,stddev=0.1))
    w2=tf.Variable(tf.random_normal(shape=[100,50],mean=0.,stddev=0.1))
    w3=tf.Variable(tf.random_normal(shape=[50,1],mean=0.,stddev=0.1))
    
    h=tf.matmul(x,w)
    h=activation(h)
    
    h1=tf.matmul(h,w1)
    h1=activation(h1)
    
    h2=tf.matmul(h1,w2)
    h2=activation(h2)
    
    
    out=tf.matmul(h2,w3)

    return out

def test_XOR():
    input_dim=2
    x=tf.placeholder(dtype=tf.float32,shape=[None,input_dim])
    y=tf.placeholder(dtype=tf.float32,shape=[None,1])    
    
    out=inference(x,input_dim)
    out=tf.nn.sigmoid(out)
    out=tf.clip_by_value(out,1e-8,1.0-1e-8)
    loss=tf.reduce_mean(tf.square(y-out))
    #loss=-tf.reduce_mean(y*tf.log(out)+(1.-y)*tf.log(1.-out))
    
    opt=tf.train.AdamOptimizer(0.001).minimize(loss)
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #XOR异或问题
    datas=[[0,0],[0,1],[1,0],[1,1]]*1000
    labels=[[1],[0],[0],[1]]*1000
    datas=np.array(datas)+np.random.normal(0,0.2,size=[4000,2])
    labels=np.array(labels)
    
    plt.scatter(datas[:,0],datas[:,1],c=labels)
    plt.show()
    
    
    y_pre=sess.run(out,feed_dict={x:datas,y:labels})
    y_pre=np.round(y_pre+0.5)
    print(y_pre)
    plt.scatter(datas[:,0],datas[:,1],c=y_pre)
    plt.show()
    
    iter_num=2000
    feed_dict={x:datas,y:labels}
    for i in range(iter_num):
        _,err=sess.run([opt,loss],feed_dict=feed_dict)
    
        if i%100==0:
            err,y_pre=sess.run([loss,out],feed_dict={x:datas,y:labels})
            print(err)
            #y_pre=np.round(y_pre+0.5)
            print(y_pre)
            plt.scatter(datas[:,0],datas[:,1],c=y_pre)
            plt.show()
    
    sess.close()
def test_Function():
    input_dim=1
    x=tf.placeholder(dtype=tf.float32,shape=[None,input_dim])
    y=tf.placeholder(dtype=tf.float32,shape=[None,1])    
    
    out=inference(x,input_dim)
    
    loss=tf.reduce_mean(tf.square(y-out))
    
    opt=tf.train.AdamOptimizer(0.001).minimize(loss)
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #训练数据：平方函数
    datas=np.array(range(-1000,1000))#np.random.rand(1000,1)
    datas=np.reshape(datas,[2000,1])/1000
    labels=np.power(datas,3)
    print(datas.shape)
    plt.plot(datas,labels,c='red')
    plt.show()
    
    
    y_pre=sess.run(out,feed_dict={x:datas,y:labels})
    print(y_pre)
    plt.plot(datas,labels,c='red')
    plt.plot(datas,y_pre,c='green')
    plt.show()
    
    iter_num=2000
    ids=random.sample(range(-1000,1000),1000)
    feed_dict={x:datas[ids],y:labels[ids]}
    for i in range(iter_num):
        _,err=sess.run([opt,loss],feed_dict=feed_dict)
    
        if i%100==0:
            err,y_pre=sess.run([loss,out],feed_dict={x:datas,y:labels})
            print(err)
            plt.plot(datas,labels,c='red')
            plt.plot(datas,y_pre,c='green')
            plt.show()
    
    sess.close()

if __name__=='__main__':
    #test_Function()
    test_XOR()