#coding:utf-8
import tensorflow as tf
import numpy as np
from pprint import pprint

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"

'''
RNNCells
'''
'''
1. tf.contrib.rnn.BasicRNNCell: 最基本的RNNCell
属性：
    * output_size
    * state_size
方法：
    (1) __init__(num_units,input_size=None,activation=tf.tanh,reuse=None)
    (2)__call__(inputs,state,scope=None):
        最基本的RNN: output=new_state=act(W*input+U*state+B)
    (3)zero_state(batch_size,dtype): 返回填充值为0的state tensor。
        * batch_size: int, float or unit
        * dtype:
        返回：
            * 如果`state_size`是int或TensorShape, 返回`N-D`tensor,shape为[batch_size,state_size],填充值为0.
            * 如果`state_size`是nested list or tuple, 返回a nested list or tuple(相同结构的)2-D tensors,每个tensor的shape为[batch_size,s]
'''
batch_size=16
input_dim=5
data=np.random.rand(batch_size,5)
tf.reset_default_graph()
inputs=tf.placeholder(dtype=tf.float32,shape=[batch_size,input_dim])
#创建RNNCell
cell=tf.contrib.rnn.BasicRNNCell(num_units=3,input_size=None)
with tf.variable_scope("basic_rnn_cell") as scope:
    '''
        * RNN的参数在第一次调用__call__的时候创建,并使用tf.get_variable()方法进行共享。
        * 第二次再调用__call__时，要使用reuse.    
    '''
    state=cell.zero_state(batch_size,dtype=tf.float32)
    output=cell(inputs,state,scope=scope)
    #获取rnn的参数
    vs=tf.get_variable_scope()
    vs.reuse_variables()
    weights=tf.get_variable(_WEIGHTS_VARIABLE_NAME,shape=None)
    
sess=tf.Session()
sess.run(tf.global_variables_initializer())
out,states=sess.run(output,feed_dict={inputs:data})
print(out)
print(states)
print(out==states)
print(sess.run(weights))
pprint(["可训练的参数：",tf.trainable_variables()])
sess.close()
'''
2. tf.contrib.rnn.BasicLSTMCell:基本的LSTM循环神经网络的Cell
属性：
    * output_size
    * state_size
方法：
    (1)__init__(num_units,forget_bias=1.0,input_size=None,state_size=None,state_is_tuple=True,activation=tf.tanh,reuse=None)
    初始化LSTM Cell
    参数：
        * num_units: int, LSTM cell的输出units的个数，输出维度
        * forget_bias: float, 加到forget gate的bias.
        * input_size: 过时，已经弃用。
        * state_is_tuple: 如果True, 接收和返回的states为两个tuples：`c_state,m_state`;
                          如果False, 它们会被concatenated along the column axis, 以后会被弃用。
        * activation: 激活函数
        * reuse: 是否reuse已存在的scope的变量，如果不是True, 并且sope中已经有了给定的变量就会报错。
    (2)__call__(inputs,state,scope=None):
        调用BasicLSTMCell,调用__call__时生成参数, LSTMCell参数weight的shape为: [input_dim+state_dim,num_units*4], state_dim=num_units
        参数：
            * inputs: 2-D tensor, shape为[batch_size,input_size]
            * state: if self.state_size is an integer, this should be a 2-D Tensor with shape [batch_size x self.state_size]. Otherwise, if self.state_size is a tuple of integers, this should be a tuple with shapes [batch_size x s] for s in self.state_size.
            * scope: VariableScope for the created subgraph; defaults to class name.
    (3)zero_state(batch_size,dtype)
'''
cell=tf.contrib.rnn.BasicLSTMCell(num_units=3,forget_bias=1.0,state_is_tuple=True)
with tf.variable_scope("lstm_cell") as scope:
    '''
        * RNN的参数在第一次调用__call__的时候创建,并使用tf.get_variable()方法进行共享。
        * 第二次再调用__call__时，要使用reuse.   
        * LSTMCell参数weight的shape为[input_dim+state_dim,num_units*4], state_dim=num_units
    '''
    initial_state=cell.zero_state(batch_size,dtype=tf.float32)
    outputs,states=cell(inputs,initial_state,scope=scope)
    #获取rnn的参数
    vs=tf.get_variable_scope()
    vs.reuse_variables()
    weights=tf.get_variable(_WEIGHTS_VARIABLE_NAME,shape=None)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
out,states=sess.run([outputs,states],feed_dict={inputs:data})
w=sess.run(weights)
print(out)
print(states)
print(out==states)
pprint(["可训练的参数：",tf.trainable_variables()])
print(w.shape)

'''
3. tf.contrib.rnn.GRUCell: Gated Recuurent Unit cell
'''

'''
4. tf.contrib.rnn.LSTMCell: Long short-term memory unit(LSTM) recurrent network cell
方法：
    (1) __init__(num_units,input_size=None,use_peepholes=False,cell_clip=None,initialize=None,num_proj=None,
                proj_clip=None,num_unit_shards=None,num_proj_shards=None,forget_bias=1.0,state_is_tuple=True,activation=tf.tanh,reuse=None)
'''

'''
5. tf.contrib.rnn.MultiRNNCell: 多层RNNCell
多层RNNCell,前一层的输出作为后一层的输入。
方法：
    (1) __init__(cells,state_is_tuple=True)
    参数：
        * cells: list of RNNCells
        * state_is_tuple: 如果True, 接收和返回n-tuples, n=len(cells)。如果为False, 拼接states。
    (2)__call__(inputs,state,scope=None)
    参数：
        * inputs： 输入2-d tensor,shape为[batch_size,input_size]
        * state: n个cell的state元组。
'''