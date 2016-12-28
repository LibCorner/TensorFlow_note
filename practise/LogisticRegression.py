# -*- coding: utf-8 -*-
#LogisticRegression
import tensorflow as tf
import mnist_data
import numpy as np

class LogisticRegression(object):
    def __init__(self,input,input_dim=784,output_dim=10):
        #定义输入
        #x=tf.placeholder('float',[None,input_dim])
        
        #定义权重的偏置
        W=tf.Variable(tf.random_uniform([784,10]))
        b=tf.Variable(tf.zeros([10]))
        
        #定义计算过程
        out=tf.matmul(input,W)+b
        
        self.p_y_given_x=tf.nn.softmax(out)
        #概率最大的维
        self.y_pred=tf.arg_max(self.p_y_given_x,dimension=1)
        self.input=input
    
    def cross_entropy(self,y):
        return -tf.reduce_mean(y*tf.log(self.p_y_given_x))

def train(learning_rate=0.1):
    #加载训练数据
    train_set,valid_set,test_set=mnist_data.load_data()
    
    train_X=train_set[0]
    train_y=train_set[1]
    y_=np.zeros((train_y.shape[0],10))
    y_[np.arange(train_y.shape[0]),train_y]=1
    
    #计算过程
    x=tf.placeholder('float',[None,784])
    y=tf.placeholder('float',[None,10])
    logistic=LogisticRegression(x,784,10)
    #y=logistic.output
    init=tf.initialize_all_variables()
    
    #损失函数
    loss=logistic.cross_entropy(y)
    #优化    
    opitimizer=tf.train.GradientDescentOptimizer(learning_rate)
    train=opitimizer.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            r=sess.run([train,loss],feed_dict={x:train_X,y:y_})
            print  r[1]

train()
