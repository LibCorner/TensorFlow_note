#coding:utf-8
import tensorflow as tf
from tensorflow.contrib.seq2seq import CustomHelper,TrainingHelper
import numpy as np
'''
class CustomHelper(Helper):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, initialize_fn, sample_fn, next_inputs_fn):
    """Initializer.

    Args:
      initialize_fn: callable that returns `(finished, next_inputs)`
        for the first iteration.
      sample_fn: callable that takes `(time, outputs, state)`
        and emits tensor `sample_ids`.
      next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
        and emits `(finished, next_inputs, next_state)`.
    """
    self._initialize_fn = initialize_fn
    self._sample_fn = sample_fn
    self._next_inputs_fn = next_inputs_fn
    self._batch_size = None

  @property
  def batch_size(self):
    if self._batch_size is None:
      raise ValueError("batch_size accessed before initialize was called")
    return self._batch_size

  def initialize(self, name=None):
    with ops.name_scope(name, "%sInitialize" % type(self).__name__):
      (finished, next_inputs) = self._initialize_fn()
      if self._batch_size is None:
        self._batch_size = array_ops.size(finished)
    return (finished, next_inputs)

  def sample(self, time, outputs, state, name=None):
    with ops.name_scope(
        name, "%sSample" % type(self).__name__, (time, outputs, state)):
      return self._sample_fn(time=time, outputs=outputs, state=state)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with ops.name_scope(
        name, "%sNextInputs" % type(self).__name__, (time, outputs, state)):
      return self._next_inputs_fn(
          time=time, outputs=outputs, state=state, sample_ids=sample_ids)
'''


class MyTrainingHelper(TrainingHelper):
    def __init__(self, embedding,bias,inputs, sequence_length, time_major=False, name=None):
        super(MyTrainingHelper,self).__init__(inputs,sequence_length,time_major=time_major,name=name)
        self.embedding=embedding
        self.bias=bias
        
    
    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
          sample_ids = tf.cast(
              tf.argmax(tf.matmul(outputs,self.embedding,transpose_b=True)+self.bias, axis=-1), tf.int32)
          return sample_ids
    def next_inputs(self,time,outputs,state,name=None,**unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with tf.name_scope(name, "TrainingHelperNextInputs",
                            [time, outputs, state]):
          next_time = time + 1
          finished = (next_time >= self._sequence_length)
          all_finished = tf.reduce_all(finished)
          def read_from_ta(inp):
            return inp.read(next_time)
          next_inputs = tf.cond(
              all_finished, lambda: self._zero_inputs,
              lambda:outputs)
          return (finished, next_inputs, state)

tf.reset_default_graph()
data=np.random.randint(low=0,high=1000,size=[1000,10])
label_data=data[:,:5]

inputs=tf.placeholder(dtype=tf.int32,shape=[None,10])
labels=tf.placeholder(dtype=tf.int32,shape=[None,5])

embedding=tf.Variable(tf.truncated_normal(shape=[1000,50],stddev=0.1,mean=0.0),dtype=tf.float32)
bias=tf.Variable(tf.zeros(shape=[1000]))
embeded=tf.nn.embedding_lookup(embedding,inputs)
with tf.variable_scope("encoder") as scope:
    cell=tf.contrib.rnn.BasicLSTMCell(num_units=50)
    #encoded_outputs,encoded_states=tf.contrib.rnn.static_rnn(cell,inputs=embeded,scope=scope)
    encoded_outputs,encoded_states=tf.nn.dynamic_rnn(cell,inputs=embeded,sequence_length=None,initial_state=None,dtype=tf.float32,time_major=False,parallel_iterations=1)
    print("encoded_states:",encoded_states)
    
with tf.variable_scope("decoder") as scope:

    cell1=tf.contrib.rnn.BasicLSTMCell(num_units=50)
    cell2=tf.contrib.rnn.BasicLSTMCell(num_units=50)
    #第二层的LSTM的初始状态
    initial_state=cell2.zero_state(batch_size=tf.shape(encoded_states[0])[0],dtype=tf.float32)
    #使用多层RNNCell
    cell=tf.contrib.rnn.MultiRNNCell([cell1,cell2],state_is_tuple=True)
    
    zero_inputs=tf.zeros_like(encoded_outputs)
    seq_len=tf.ones([tf.shape(inputs)[0],],dtype=tf.int32)*5
    
    helper=MyTrainingHelper(embedding=embedding,bias=bias,inputs=zero_inputs,sequence_length=seq_len,time_major=False)
    
    decoder=tf.contrib.seq2seq.BasicDecoder(cell,helper,initial_state=(encoded_states,initial_state),output_layer=None)
    [outputs,ids],states=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=False)
    
    print(type(outputs))
    print(type(states))
    print("decode outputs:",outputs)

def sampled_loss(labels,inputs,embedding,bias):
    labels=tf.expand_dims(labels,1) #需要变成2-D的
    loss=tf.nn.sampled_softmax_loss(embedding,bias,labels=labels,inputs=inputs,num_sampled=10,num_classes=1000)
    return loss 
    
loss=tf.contrib.seq2seq.sequence_loss(outputs,targets=labels,weights=tf.ones_like(labels,dtype=tf.float32),softmax_loss_function=lambda x,y:sampled_loss(x,y,embedding,bias))
opt=tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in range(10):
        _,l,out,idx=sess.run([opt,loss,outputs,ids],feed_dict={inputs:data,labels:label_data})
        print("loss",l)