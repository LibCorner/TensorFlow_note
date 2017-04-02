#coding:utf-8
import tensorflow as tf
import numpy as np


data=np.random.rand(100,10)
label=np.random.rand(100,1)
batch=tf.train.shuffle_batch([data,label],batch_size=10,capacity=100,min_after_dequeue=10,enqueue_many=True)


sess=tf.Session()
print(len(batch))
print(batch)
