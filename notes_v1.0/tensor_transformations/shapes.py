#coding:utf-8
import tensorflow as tf

'''
1. 获取tensor的shape: tf.shape(input,name=None,out_type=tf.int32)
参数：
    * input: Tensor或SparseTensor
返回：
    * Tensor, type为out_type
2. 获取tensor的size: tf.size(input,name=None,out_type=tf.int32)
返回tensor的size
3. 获取tensor的rank(秩): tf.rank(input,name=None)
4. 获取多个tensors的shape: tf.shape_n(input,out_type=None,name=None)
参数：
    * input: 至少有一个tensor的list
'''
a=tf.Variable([[1,2,4],[3,4,5]])
s=tf.shape(a)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
out=sess.run(s)
print("shape:",out)

size=tf.size(a)
out=sess.run(size)
print("size:",out)

rank=tf.rank(a)
out=sess.run(rank)
print("rank:",out)

shape_n=tf.shape_n([a])
out=sess.run(shape_n)
print("shape_n:",out)

'''
5. 改变tensor的shape: tf.reshape

6. 扩展维度： tf.expand_dims

7. 去掉size为1的维度：tf.squeeze(input,axis=None,name=None,squeeze_dims=None)
参数：
    * input: Tensor
    * axis: a list of ints, 可选。要去掉的维度。
    * name：
    * squeeze_dims: 过时的参数，同axis。
8. Broadcasts parameters : tf.meshgrid()
'''