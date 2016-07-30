#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import math

NUM_CLASSES=10
IMAGE_SIZE=28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(image,hidden1_unit,hidden2_unit):
    '''
    参数：
        image: 输入的图像
        hidden1_unit: 第一个隐藏层的神经元个数
        hidden2_unit:第二个隐藏层的神经元个数
    返回：
        输出结果
    '''
    #定义每一层的参数和计算过程
    with tf.name_scope("hidden1"):
        w=tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden1_unit], stddev=1.0/math.sqrt(IMAGE_PIXELS)),name='w')
        b=tf.Variable(tf.zeros([hidden1_unit]))
        hidden1=tf.nn.relu(tf.matmul(image,w)+b)
    
    with tf.name_scope("hidden2"):
        w=tf.Variable(tf.truncated_normal([hidden1_unit,hidden2_unit],stddev=1.0/math.sqrt(hidden1_unit)),name='w')
        b=tf.Variable(tf.zeros([hidden2_unit]))
        hidden2=tf.nn.relu(tf.matmul(hidden1,w)+b)

    with tf.name_scope("softmax_linear"):
        w=tf.Variable(tf.truncated_normal([hidden2_unit,NUM_CLASSES],stddev=1.0/math.sqrt(hidden2_unit)),name='w')
        b=tf.Variable(tf.zeros([NUM_CLASSES]))
        logits=tf.matmul(hidden2,w)+b
        logits=tf.nn.softmax(logits)

    return logits

def loss(logits,labels):
    """定义损失函数"""
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels,name='xentropy')
    loss=tf.reduce_mean(cross_entropy,name='xentropy-mean')
    return loss

def training(loss,learning_rate):
    tf.scalar_summary(loss.op.name,loss)
    #优化器
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    global_step=tf.Variable(0,name='global_step',trainable=False)
    train_op=optimizer.minimize(loss,global_step=global_step)

    return train_op

def evaluation(logits,labels):
    correct=tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))
