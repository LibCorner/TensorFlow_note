#coding:utf-8
import tensorflow as tf
import numpy as np


tf.reset_default_graph()
data=np.random.randint(low=0,high=1000,size=[1000,10])
label_data=data[:,:5]

inputs=tf.placeholder(dtype=tf.int32,shape=[None,10])
labels=tf.placeholder(dtype=tf.int32,shape=[None,5])

embedding=tf.Variable(tf.truncated_normal(shape=[1000,50],stddev=0.1,mean=0.0),dtype=tf.float32)
embeded=tf.nn.embedding_lookup(embedding,inputs)
with tf.variable_scope("encoder") as scope:
    cell=tf.contrib.rnn.BasicLSTMCell(num_units=50)
    #encoded_outputs,encoded_states=tf.contrib.rnn.static_rnn(cell,inputs=embeded,scope=scope)
    encoded_outputs,encoded_states=tf.nn.dynamic_rnn(cell,inputs=embeded,sequence_length=None,initial_state=None,dtype=tf.float32,time_major=False,parallel_iterations=1)
    print("encoded_states:",encoded_states)
    
with tf.variable_scope("decoder") as scope:

    cell1=tf.contrib.rnn.BasicLSTMCell(num_units=50)
    cell2=tf.contrib.rnn.BasicLSTMCell(num_units=1000)
    #第二层的LSTM的初始状态
    initial_state=cell2.zero_state(batch_size=tf.shape(encoded_states[0])[0],dtype=tf.float32)
    #使用多层RNNCell
    cell=tf.contrib.rnn.MultiRNNCell([cell1,cell2],state_is_tuple=True)
    
    zero_inputs=tf.zeros_like(encoded_outputs)
    seq_len=tf.ones([tf.shape(inputs)[0],],dtype=tf.int32)*5
    helper=tf.contrib.seq2seq.TrainingHelper(inputs=zero_inputs,sequence_length=seq_len,time_major=False)
    #helper=tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=zero_inputs,sequence_length=seq_len,embedding=embedding,sampling_probability=1.0,time_major=False)
    #helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,start_tokens=tf.ones([tf.shape(embeded)[0]],dtype=tf.int32),end_token=0)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell,helper,initial_state=(encoded_states,initial_state),output_layer=None)
    [outputs,ids],states=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=False)
    
    print(type(outputs))
    print(type(states))
    print("decode outputs:",outputs)
    
loss=tf.contrib.seq2seq.sequence_loss(outputs,targets=labels,weights=tf.ones_like(labels,dtype=tf.float32),softmax_loss_function=None)
opt=tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in range(10):
        _,l,out,idx=sess.run([opt,loss,outputs,ids],feed_dict={inputs:data,labels:label_data})
        print("loss",l)