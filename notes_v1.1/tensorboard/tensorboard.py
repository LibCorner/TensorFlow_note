#coding:utf-8
import tensorflow as tf

'''
使用tensorboard可视化
1. 使用summary operations从Graph中收集summary data
    (1)比如，使用`tf.summary.scalar`来收集训练过程中学习速率或loss的变化
    (2)使用`tf.summary.histogram`来关联梯度、输出和网络权重, 可以查看它们的分布情况。
2. 生成summary数据：
（1）Tensorflow中的operation只有在run的时候才会起作用，其他的计算操作都不依赖summary操作，
    所以要想生成summaryies,我们需要运行所有的summary nodes.
（2）为了方便运行所有的summary ops可以使用`tf.summary.merge_all`把所有的summary ops合并成一个ops.
    然后，只需要运行合并的summary op就可以生成包含所有suumary数据的serialized Summary protobuf 对象。
（3）最后，使用`tf.summary.FileWriter`把summary数据写到磁盘中。
2. 运行命令进行可视化：
    (1). tensorboard --logdir="./graphs" --port 6006 
    (2). 打开网页： http://localhost:6006/
'''
sess=tf.Session()
a=tf.constant(2)
b=tf.constant(3)
x=tf.add(a,b)

tf.summary.scalar(name='a',tensor=a)
tf.summary.scalar(name='b',tensor=b)
tf.summary.scalar(name='x',tensor=x)

#合并所有的summary
merged=tf.summary.merge_all()


#使用TensorBoard
writer=tf.summary.FileWriter('./graphs',sess.graph)

print(sess.run(x))
summ=sess.run(merged)
writer.add_summary(summ)

#使用完后关闭writer
writer.close()

