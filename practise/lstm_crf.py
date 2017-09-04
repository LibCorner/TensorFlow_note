#coding:utf-8
import tensorflow as tf
import numpy as np


tf.reset_default_graph()

word_ids=tf.placeholder(dtype=tf.int32,shape=[None,None])
word_length=tf.placeholder(dtype=tf.int32,shape=[None])
labels=tf.placeholder(dtype=tf.int32,shape=[None,None],name="labels")

with tf.variable_scope("embedding"):
    embedding=tf.get_variable(name="embedding",shape=[10000,50])
    embeded=tf.nn.embedding_lookup(embedding,word_ids)
    
with tf.variable_scope("lstm"):
    lstm_cell_fw=tf.nn.rnn_cell.BasicLSTMCell(50,activation=tf.nn.tanh)
    lstm_cell_bw=tf.nn.rnn_cell.BasicLSTMCell(50,activation=tf.nn.tanh)
    
    (outputs_fw,outputs_bw),(final_state_fw,final_state_bw)=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                                            cell_bw=lstm_cell_bw,inputs=embeded,sequence_length=word_length,dtype=tf.float32)
    
    context=tf.concat([outputs_fw,outputs_bw],axis=-1)
    
with tf.variable_scope("output"):
    W=tf.get_variable("w",shape=[100,4])
    b=tf.get_variable('b',shape=[4])
    
    ctx=tf.reshape(context,shape=[-1,100])
    outputs=tf.matmul(ctx,W)+b
    scores=tf.reshape(outputs,shape=[-1,tf.shape(context)[1],4])
    
#CRF liklihood loss
'''
crf_log_likelihood(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params=None
)
参数：
inputs: [batch_size,max_seq_len,num_tags]
tag_indices: [batch_size,max_seq_len], 标签的索引indices, 用来计算log-likelihood
sequence_lengths: [batch_size]
transition_params: [num_tags,num_tags]，转移矩阵

返回：
log_likelihood: 
transition_params:
'''

log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(scores,labels,word_length)
#计算损失
loss=tf.reduce_mean(-log_likelihood)

#训练
optimizer=tf.train.AdamOptimizer(0.001)
train_op=optimizer.minimize(loss)

#预测
'''
tf.contrib.crf.viterbi_decode(score,transition_params)
参数：
score: [seq_len,num_tags], 只能输入一个样本
transition_param: [num_tags,num_tags]
返回：
viterbi: A [seq_len] list of integers, 包括得分最高的tag indices
viterbi_score: Viterbi 序列解码的分数， float
'''
#viterbi_sequence,viterbi_score=tf.contrib.crf.viterbi_decode(scores,transition_params)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

words=np.random.randint(0,10000,size=[128,10])
seq_length=np.random.randint(5,10,size=[128])
label=np.random.randint(0,4,size=[128,10])

for i in range(10):
    _,loss=sess.run([train_op,loss],feed_dict={word_ids:words,word_length:seq_length,labels:label})
    print(loss)
