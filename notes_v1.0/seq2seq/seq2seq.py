#coding:utf-8
import tensorflow as tf


'''
模块：tf.contrib.seq2seq
创建seq2seq模型中的decoders和loss的Ops
'''

'''
注意：解码器的RNNCell与编码器的RNNCell结构要一致，因为编码器的状态(h,c)要用来初始化解码器的状态，如果不一致就会出错。
'''


'''
1. dynamic_rnn_decoder: 
   * seq2seq模型的动态rnn解码器,由RNNCell和decoder函数具体化
   * dynamic_rnn_decoder与tf.python.ops.rnn.dynamic_rnn相似，对序列长度和输入batch大小没有assuption
   * dynamic_rnn_decoder有两个模型： 训练和推理，需要用户为每个模型分别创建decoder函数。
   * 在训练和推理中，都需要`cell`和`decoder_fn`, `cell`使用`raw_rnn`执行每个时刻的计算，`decoder_fn`允许对early stopping,output,state,next input和context建模。
   * 当训练时，用户需要提供`inputs`, 在每个时刻·inputs·的一个slice输入到`decoder_fn`中，修改和返回下一时刻的输入。
   * 训练时，需要`sequence_length`, 测试时不需要。
   * 当inference时，`input`应该是`None`,输入是从`decoder_fn`中得到的
参数：
    * cell: RNNCell的实例
    * decoder_fn: decoder函数，输入为time,cell state, cell input, cell output和context state, 
      返回early stopping向量，cell state, next input, cell output和context state。`decoder_fn`的例子可以在 decoder_fn.py中找到。
    * inputs: 解码的输入(embeded_fromat)
       (1) 如果`time_major==False`(默认)， shape必须为[batch_size, max_time, ...]
       (2) 如果`time_major==True`, inputs的shape必须是[max_time,batch_size,...]
       每个时刻输入到`cell`中的tensor的shape为[batch_size,...]
    * sequence_length:(可选)，int32/int64，shape为[batch_size],如果inputs不是None,而sequence_length是None, 会根据inputs推断最大可能的序列长度。
    * paralel_iterations: (默认：32)。并行运行的iteration的个数。这个参数以空间换时间，values>>1时，使用更多的内存，花费的时间变少，值比较小时，使用较少的内存，但是计算时间更长。
    * swap_memory: 
    * time_major: `inputs`和`outputs`的格式，使用time_major=True更加高效，因为这样可以避免在RNN开始和结束时进行tranpose。然而很多数据都是batch-major的，因此默认为False。
    * scope: 变量作用域VariableScope，默认为None.
    * name: NameScope,默认为'dynamic_rnn_decoder'
返回：
    元组(outputs,final_state,final_context_state)
    * ouputs: RNN输出的Tensor:
        如果`time_major`==False, shape为[batch_size,max_time,cell.output_size]
        如果`time_major==True`, shape为[max_time,batch_size,output_size]
    * final_state: 最终的状态，shape为[batch_size,cell.state_size]
    * final_context_state: 最后一个调用`decoder_fn`返回的context state
'''

'''
2. attention_decoder_fn_inference: 
 * `dynamic_rnn_decoder`在inference过程中使用的attentional decoder函数
 * `attention_decoder_fn_inference`返回解码函数`decoder_fn`,可以传入到`dynamic_rnn_decoder`中。
 * 更多的用法可以参考`kernel_tests/seq2seq_test.py`
 参数：
    * output_fn: 输出函数，用来把`cell_output`映射为class logits
      比如：output_fn=lambda x:layers.linear(x,num_decoder_symbols,scope=varscope)
    * ecoder_state: 编码器的状态，用来初始化`dynamic_rnn_decoder`
    * attention_keys: 用来与target states比较
    * attention_values: 用来构建context vectors.
    * attention_score_fn: 用来计算key和target states相似度的函数
    * attention_construct_fn: 用来build attention states的函数
    * embeddings: embedding matrix, shape为[num_decoder_symbols,embedding_size]
    * start_of_sequence_id: The start of sequence ID in the decoder embeddings。
    * end_of_sequence_id: he end of sequence ID in the decoder embeddings.
    * maximun_length: decode的最大time step
    * num_decoder_symbols: decoder的类别数。
    * dtype:
    * name: NameScope
'''


'''
3. attention_decoder_fn_train
* `dynamic_rnn_decoder`在training过程中使用的attentional decoder函数
* `attention_decoder_fn_train`返回解码函数`decoder_fn`,可以传入到`dynamic_rnn_decoder`中。
参数：
    * ecoder_state: 编码器的状态，用来初始化`dynamic_rnn_decoder`
    * attention_keys: 用来与target states比较
    * attention_values: 用来构建context vectors.
    * attention_score_fn: 用来计算key和target states相似度的函数
    * attention_construct_fn: 用来build attention states的函数
    * name: NameScope
'''


'''
4. prepare_attention: 为attention准备keys/values/functions
参数：
    * attention_states: hidden states to attend over.
    * attenion_option: 如何计算attention, `luong`或`bahdanau`:
        (1) `bahdanau`: additive （Bahdanau et al., ICLR'2015）
        (2) `luong`: multiplicative (Luong et al., EMNLP'2015)
    * num_units: hidden state dimension.
    * reuse: 是否reuse variable scope`
返回：
    * attention_keys: 用来与target states比较
    * attention_values: 用来构建context vectors.
    * attention_score_fn: 用来计算key和target states相似度的函数：
        （1）attention_score_fn(query,keys,values):
            * query: shape=[batch_size,num_units], 上一时刻的输出 h_{t-1}
            * keys: shape=[batch_size,attention_length,num_units]，values经过线性转换后的序列， keys=values*W.
            * values: shape=[batch_size,attention_length, num_units]，要attention的序列, values=attention_states
            返回：
                context_vector: shape=[batch_size,num_units]
    * attention_construct_fn: 用来build attention states的函数:
        (1) 参数:
                * name: 变量名字
                * num_nuits: 隐层维度
                * attention_score_fn: 计算key与target states相似度的函数
        （2）返回：
                * attention向量: shape=[batch_size,num_units]
                * attention的计算： attention_query与attention_score_fn返回的context_vector拼接，然后经过一个线性变换
'''

'''
5. sequence_loss: 一个logits序列的加权交叉熵损失（per example）
参数：
    * logits: 3-D, shape=[batch_size,sequence_length,num_decoder_symbols], num_decoder_symbols为decoder的维度，可以是类别个数，也可以是词向量的维度（需要使用负采样的softmax_loss_function）
    * targets: 2-D, shape=[batch_size,sequence_length], dtype为int,目标序列的index
    * weights: 2-D, shape=[batch_size,sequence_length],dtype为float
    * softmax_loss_function: Function(inputs-batch,labels-batch)
'''

""" 
6. simple_decoder_fn_inference:
    * 返回 `dynamic_rnn_decoder`在inference过程中使用的decoder函数
Decoder function used in the `dynamic_rnn_decoder` with the purpose of
inference.

The main difference between this decoder function and the `decoder_fn` in
`simple_decoder_fn_train` is how `next_cell_input` is calculated. In this
decoder function we calculate the next input by applying an argmax across
the feature dimension of the output from the decoder. This is a
greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
use beam-search instead.

Args:
  time: positive integer constant reflecting the current timestep.
  cell_state: state of RNNCell.
  cell_input: input provided by `dynamic_rnn_decoder`.
  cell_output: output of RNNCell.
  context_state: context state provided by `dynamic_rnn_decoder`.

Returns:
  A tuple (done, next state, next input, emit output, next context state)
  where:

  done: A boolean vector to indicate which sentences has reached a
  `end_of_sequence_id`. This is used for early stopping by the
  `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
  all elements as `true` is returned.

  next state: `cell_state`, this decoder function does not modify the
  given state.

  next input: The embedding from argmax of the `cell_output` is used as
  `next_input`.

  emit output: If `output_fn is None` the supplied `cell_output` is
  returned, else the `output_fn` is used to update the `cell_output`
  before calculating `next_input` and returning `cell_output`.

  next context state: `context_state`, this decoder function does not
  modify the given context state. The context state could be modified when
  applying e.g. beam search.
  """
'''
7. simple_decoder_fn_train: `dynamic_rnn_decoder`在训练时使用的简单decoder

参数：
    * encoder_sate: encoded state，用来初始化`dynamic_runn_decoder`
    * name: NameScope
返回：一个解码器函数
'''