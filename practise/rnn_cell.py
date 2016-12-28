#-*- coding:utf-8-*-
import tensorflow as tf
import numpy as np

batch_size=5

x=tf.placeholder(tf.float32,shape=[5,10])
data=np.random.rand(5,10)
#定义rnn_cell
cell=tf.nn.rnn_cell.BasicRNNCell(10)
state=cell.zero_state(batch_size,tf.float32)
#计算rnn_cell的输出
output,state=cell(x,state)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(output,feed_dict={x:data})

#定义GRUCell
gru_cell=tf.nn.rnn_cell.GRUCell(num_units=10,activation=tf.tanh)
state=cell.zero_state(batch_size,tf.float32)
#计算输出和状态
output,state=gru_cell(x,state)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run([output,state],feed_dict={x:data})

#定义BasicLSTMCell
lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=10,forget_bias=1.0,state_is_tuple=True,activation=tf.tanh)
state=lstm_cell.zero_state(batch_size,tf.float32)
#计算输出和状态
output,state=lstm_cell(x,state)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run([output,state],feed_dict={x:data})
