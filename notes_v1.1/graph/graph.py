#coding:utf-8
import tensorflow as tf
'''
tf.Graph: 
1. Graph包含了表示计算单元的tf.Operation 对象和表示数据单元的tf.Tensor对象。
2. 通常会有一个默认的`Graph`，并可以通过`tf.get_default_graph`来访问
3. 定义一个新的`Operation`就会把这个operation加到默认图中。
'''

c=tf.constant(4.0)
assert c.graph is tf.get_default_graph()

'''
4. 可以使用`tf.Graph()`定义一个新的Graph
注意： 这个class对于图的构建不是线程安全的，所有的operations应该在当个线程创建，或者使用同步机制(synchronization)。
'''
g=tf.Graph()
with g.as_default():
    c=tf.constant(30.0)
    assert c.graph is g
    
'''
5. 多个图需要多个sessions, 默认情况下每个都会试图使用所有的可用资源。
6. 不使用python/numpy传递数据时，无法在多个图之间传输数据
7. 最好在一个图中使用多个不相连的子图。
8. 不要混用默认图和用户定义的图
'''
sess=tf.Session(graph=g) #session中指定图
sess.run(c)


'''
方法：
1. add_to_collection(name,value): 把`value` 存储到名为`name`的collection中
注意： collections不是set, 可以多次把同一个value加到collection里

2. add_to_collections(names,value): names可以是任何可迭代对象.

3. as_default(): 返回一个把当前Graph作为默认图的context manager。
该方法要与`with`代码块配合使用： with g.as_default()

4. clear_collection(name): 清空名为name的collection里的values

'''