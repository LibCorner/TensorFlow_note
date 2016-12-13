#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

batch_size=5
length=10
unit_num=20
idx=np.random.randint(low=0,high=1000,size=(batch_size,length))
embedding=np.random.rand(1000,50)
#embbedding
inputs=tf.nn.embedding_lookup(embedding,idx)
#unpack成sequence
inputs=tf.unpack(inputs,num=length,axis=1)
#定义lstm_cell
lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(unit_num,forget_bias=1.0,state_is_tuple=True,activation=tf.tanh)
#获取初始状态
initial_state=lstm_cell.zero_state(batch_size,tf.float64)
#rnn
states,output=tf.nn.rnn(lstm_cell,inputs,initial_state)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run([states,output])
#rnn简单的内部实现
inputs=tf.nn.embedding_lookup(embedding,idx)
state=initial_state
outputs=[]
with tf.variable_scope("RNN",reuse=True):
    for time_step in range(length):
        if time_step>0: tf.get_variable_scope().reuse_variables()
        (cell_output,state)=lstm_cell(inputs[:,time_step,:],state)
        outputs.append(cell_output)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(outputs)
