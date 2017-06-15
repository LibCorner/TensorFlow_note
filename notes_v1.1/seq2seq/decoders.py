#coding:utf-8
import tensorflow as tf
import numpy as np
'''
1. tf.contrib.seq2seq.dynamic_decode
参数：
    * decoder: A `Decoder`对象。
    * output_time_major: boolean.默认为`False`(batch major),如果True,outputs返回time major tensors(速度更快)，否则返回batch major tensors.
    * impute_finished: boolean.  If True, then states for batch entries which are marked as finished get copied through and the corresponding outputs get zeroed out. This causes some slowdown at each time step, but ensures that the final state and outputs have the correct values and that backprop ignores time steps that were marked as finished.
    * maximum_iterations: int32, decoding允许的最大step。
    * parallel_iterations: 传递给`tf.while_loop`的参数。
    * swap_memory: 传递给`tf.while_loop`的参数。
    * scope: 可选的scope变量。
返回：
    (final_outputs,final_state):
        * final_outputs: `BasicDecoderOutput`类
        * final_state: LSTMStateTuple
'''

'''
2. tf.contrib.seq2seq.Decoder: RNN Decoder的抽象接口
属性：
    * batch_size: 
    * output_dtype: 
方法：
    (1) initialize: 在decoding迭代前调用,返回(finished,first_inputs,initial_state)
    (2) setp(time,inputs,state,name=None): 每个解码step调用
    参数：
        * time: int32
        * inputs: 该时刻输入的tensor
        * state: 上一时刻的state
        * name: name scope
    返回：
        (outputs,nex_state,next_inputs,finished)
3. tf.contrib.seq2seq.BasicDecoder: 基本的Decoder
方法：
    (1) __init__(cell,helper,initial_state,output_layer=None): 初始化
        * cell: RNNCell实例。
        * helper: `Helper`实例。
        * initial_state: 
        * output_layer: 可选，`tf.layers.Layer`实例，比如tf.layers.Dense
4. tf.contrib.seq2seq.BasicDecoderOutput: decode的输出
属性：
    * rnn_output: Alias for field number 0
    * sample_id: Alias for field number 1.
方法：
    __new__(_cls,rnn_output,sample_id):创建新的`BasicDecoderOutput(rnn_output,sample_id)`实例。

'''

'''
5. Decoder Helpers
(1) tf.contrib.seq2seq.Helper: Helper接口， Helper实例会被SamplingDecoder使用
属性：
    * batch_size
方法：
    * initialize(name=None): 返回(initial_finished,initial_inputs)
    * next_inputs(time,outputs,state,sample_ids,name=None): 返回(finiesed,next_inputs,next_state)
    * sample(time,outputs,state,name=None): 返回sample_ids
(2) tf.contrib.seq2seq.CustomHelper
方法：
    * __init__(initialize_fn,sample_fn,next_inputs_fn):
        1. initalize_fn: 返回(finished,next_inputs)的函数，第一次迭代时使用。
        2. sample_fn: 参数为(time,outputs,state),生成sample_ids的函数
        3. next_inputs_fn: callable that takes (time, outputs, state, sample_ids) and emits (finished, next_inputs, next_state)
(3)tf.contrib.seq2seq.GreedyEmbeddingHelper
使用outpu的argmax作为ids传给embedding层来生成下一个input.
方法：
    * __init__(embedding,start_tokens,end_token)
        Args:
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.
          
(4) tf.contrib.seq2seq.ScheuledEmbeddingTrainingHelper
Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      sampling_probability: A 0D `float32` tensor: the probability of sampling
        categorically from the output ids instead of reading directly from the
        inputs.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      seed: The sampling seed.
      scheduling_seed: The schedule decision rule sampling seed.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sampling_probability` is not a scalar or vector.
    """

(5) tf.contrib.seq2seq.TrainingHelper
* A helper for use during training.  Only reads inputs.
* Returned sample_ids are the argmax of the RNN output logits.


'''

tf.reset_default_graph()
data=np.random.randint(low=0,high=1000,size=[1000,10])

inputs=tf.placeholder(dtype=tf.int32,shape=[None,10])
embedding=tf.Variable(tf.truncated_normal(shape=[1000,50],stddev=0.1,mean=0.0),dtype=tf.float32)
embeded=tf.nn.embedding_lookup(embedding,inputs)
with tf.variable_scope("encoder") as scope:
    cell=tf.contrib.rnn.BasicLSTMCell(num_units=20)
    #encoded_outputs,encoded_states=tf.contrib.rnn.static_rnn(cell,inputs=embeded,scope=scope)
    encoded_outputs,encoded_states=tf.nn.dynamic_rnn(cell,inputs=embeded,sequence_length=None,initial_state=None,dtype=tf.float32,time_major=False)
    print("encoded_states:",encoded_states)
    
with tf.variable_scope("decoder") as scope:
    '''
    cell1=tf.contrib.rnn.BasicLSTMCell(num_units=20)
    cell2=tf.contrib.rnn.BasicLSTMCell(num_units=1000)
    #第二层的LSTM的初始状态
    initial_state=cell2.zero_state(batch_size=tf.shape(encoded_states[0])[0],dtype=tf.float32)
    #使用多层RNNCell
    cell=tf.contrib.rnn.MultiRNNCell([cell1,cell2],state_is_tuple=True)
    
    zero_inputs=tf.zeros_like(encoded_outputs)
    seq_len=tf.ones([tf.shape(inputs)[0],],dtype=tf.int32)*5
    helper=tf.contrib.seq2seq.TrainingHelper(inputs=zero_inputs,sequence_length=seq_len,time_major=False)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell,helper,initial_state=(encoded_states,initial_state),output_layer=None)
    outputs,states=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=False)
    '''
    cell=tf.contrib.rnn.BasicLSTMCell(num_units=20)
    att_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=20,memory=encoded_outputs)
    att_cell=tf.contrib.seq2seq.DynamicAttentionWrapper(cell=cell,attention_mechanism=att_mechanism,attention_size=20)
    att_states=tf.contrib.seq2seq.DynamicAttentionWrapperState(cell_state=encoded_states,attention=tf.reduce_mean(encoded_outputs,axis=1))
    
    zero_inputs=tf.zeros_like(encoded_outputs)
    seq_len=tf.ones([tf.shape(inputs)[0],],dtype=tf.int32)*5
    helper=tf.contrib.seq2seq.TrainingHelper(inputs=zero_inputs,sequence_length=seq_len,time_major=False)
    decoder=tf.contrib.seq2seq.BasicDecoder(att_cell,helper,initial_state=att_states,output_layer=None)
    outputs,states=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=False)
    print(type(outputs))
    print(type(states))
    print("decode outputs:",outputs)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out,ids=sess.run(outputs,feed_dict={inputs:data})
