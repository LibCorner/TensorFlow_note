#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

unit_num=10
batch_size=5
length=10
dim=50
vocab_size=1000

ids=np.random.randint(low=0,high=1000,size=(batch_size,length))
embedding=np.random.rand(1000,dim)
x=tf.nn.embedding_lookup(embedding,ids)
#unstack,变成sequence
#x=tf.unstack(x,num=length,axis=1)
x=tf.unpack(x,num=length,axis=1)
#定义rnn_cell
rnn_cell=tf.nn.rnn_cell.BasicRNNCell(unit_num)
initial_state=rnn_cell.zero_state(batch_size,tf.float64)
#rnn
outputs,state=tf.nn.rnn(rnn_cell,x,initial_state)

data=np.random.rand(20,length,dim)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    o=sess.run([outputs,state])
    print o
