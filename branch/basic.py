# -*- coding: utf-8 -*-
#basic
import tensorflow as tf
import numpy as np

'''
TensorFlow使用图(graph)来表示计算任务，
在会话（session）的上下文（context）中执行图
使用tensor表示数据
通过变量（Variable）维护状态。
使用feed和fetch可以为任意的操作赋值或从中获取数据。
'''
## 构建阶段，op的执行步骤被描述成一个图。
## 执行阶段，使用session执行图中的op。

## 构建图
## 1.首先要构建源op, 源op不需要任何输入, 例如常量（Constant）
## 源op的输出被传递给其他op
'''
python库中，op构造器的返回值代表该op的输出，这些返回值可以传递给其他op作为输入。
tensorflow python中有一个默认图（default graph）,op构造器可以为其增加节点。
'''

#创建一个常量op,产生一个1x2矩阵，这个op作为一个节点加到默认图中
#构造器的返回值代表该常量op的返回值
matrix1=tf.constant([[3.,3.]])

#创建另外一个常量op,产生一个2x1的矩阵
matrix2=tf.constant([[2.],[2.]])

#创建一个矩阵乘法matmul op, 把matrix1和matrix2作为输入
#返回product代表矩阵乘法的结果
product=tf.matmul(matrix1,matrix2)

#启动一个默认图
sess=tf.Session()

#调用sess的run() 方法来执行矩阵乘法op, 传入'product'作为该方法的参数。
#上面提到，‘product’代表矩阵乘法op的输出，传入它是向方法表明，我们希望取回矩阵乘法op的输出。
#
#整个执行过程是自动化的，会话负责传递op所需要的全部输入。op通常是并发执行的。
#
#函数'run(product)'触发了图中三个op的执行。
#
#返回值`result`是一个numpy `ndarray`对象。
result=sess.run(product)
print result

#任务完成，关闭会话
sess.close()

'''
Session对象在使用完后需要关闭以释放资源，除了显式调用close外，也可以使用with代码块来自动完成关闭动作。
'''
with tf.Session() as sess:
    result=sess.run([product])
    print result
    
## Fetch
'''
为了取回操作的输出内容，可以在使用Session对象的run() 调用执行图时，传入多个tensor来取回多个tensor值
'''
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
intermed=tf.add(input2,input3)
mul=tf.mul(input1,intermed)

with tf.Session() as sess:
    result=sess.run([mul,intermed])
    print result

## Feed
'''
TensorFlow提供了feed机制，该机制可以临时替代图中的任意操作中的tensor。
你可以提供feed数据作为run() 调用的参数。feed只在调用它的方法内有效，方法结束，feed就会消失。
标记的方法是使用tf.placeholder()创建占位符。
'''
input1=tf.placeholder(tf.types.float32)
input2=tf.placeholder(tf.types.float32)
output=tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run([output],feed_dict={input1:[7.],input2:[2.]})

