###  tf.contrib.seq2seq.AttentionWrapper
#### 1. `__init__`

	__init__(
		cell,  
		attention_mechanism,
		attention_layer_size=None,
		alignment_history=False,
		cell_input_fn=None,
		output_attention=True,
		initial_cell_state=None,
		name=None)

> 注意： 如果使用`BeamSearchDecoder`, 必须保证:
> > * encoder ouput 要使用`tf.contrib.seq2seq.tile_batch` (不是`tf.tile`)tiled 成 `beam_width`。
> >* 传给`zero_state`方法的`batch_size`参数的等于`true_batch_size* beam_width`
> > * 使用`zero_state`创建的inital state包含了正确tiled了的encoder的final state的`cell_state` 值。
> 
> 比如：

	tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
	    encoder_outputs, multiplier=beam_width)
	tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
	    encoder_final_state, multiplier=beam_width)
	tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
	    sequence_length, multiplier=beam_width)
	attention_mechanism = MyFavoriteAttentionMechanism(
	    num_units=attention_depth,
	    memory=tiled_inputs,
	    memory_sequence_length=tiled_sequence_length)
	attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
	decoder_initial_state = attention_cell.zero_state(
	    dtype, batch_size=true_batch_size * beam_width)
	decoder_initial_state = decoder_initial_state.clone(
	    cell_state=tiled_encoder_final_state)

> 参数:
> > * cell: RNNCell
> > * attention_mechanism: A list of `AttentionMechanism` 实例或者一个实例。
> >* attention_layer_size: 整数或者整数列表，attention输出层的depth。如果是None, 使用context作为attention, 否则，把context和cell的输出输入到attention层生成attention.
> >* alignment_history: boolean, 是否存储alignment history。
> >*  cell_input_fn: A callable. 默认： `lambda inputs,attention: array_ops.concat([inputs,attention],-1)`
> >* output_attention: bool. 如果为True, 每步的输出是attention值， 如果为False, 每步的输出是cell的ouput.
> >*  initial_cell_state: 当用户调用`zero_state()`时返回的初始状态。

### tf.contrib.seq2seq.tile_batch

	tile_batch(
		t,
		multipier,
		name=None)
>参数：
>>* t: `Tensor`, shape：`[batch_size, ...]`
>>* multiplier: int, tile的次数
>>* name: name scope
>
> 返回
> > shape为`[batch_size*multiplier, ...]`的Tensor

### tf.sequence_mask

	sequence_mask(
		lengths,
		maxlen=None,
		dtype=tf.bool,
		name=None
	)

返回一个mask tensor， 表示每个cell的前N个位置。

如果`lengths`的shape为`[d_1,d_2, ..., d_n]`, 得到的tensor`mask`的dtype 为`dtype`，shpae为`[d_1,d_2, ... , d_n, maxlen]`, 值为：

	mask[i_1,i_2,..., i_n, j]=(j<lengths[i_1, i_2, ..., i_n])

比如：

	tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
	                                #  [True, True, True, False, False],
	                                #  [True, True, False, False, False]]
	
	tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
	                                  #   [True, True, True]],
	                                  #  [[True, True, False],
	                                  #   [False, False, False]]]

### tf.nn.dynamic_rnn

	dynamic_rnn(
	    cell,
	    inputs,
	    sequence_length=None,
	    initial_state=None,
	    dtype=None,
	    parallel_iterations=None,
	    swap_memory=False,
	    time_major=False,
	    scope=None
	)

使用RNNCell `cell` 创建一个循环神经网络。
> 返回：
> > * (outputs, state) 
> >* outputs: RNN输出的Tensor.
> >如果time_major=False(default), shape为`[batch_size,max_time,cel.output_size]`; 如果time_major=True, shape为`[max_time,batch_size,cell.output_size]`。
> * state: 最终的state. 

例子：

	# create a BasicRNNCell
	rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
	
	# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
	
	# defining initial state
	initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
	
	# 'state' is a tensor of shape [batch_size, cell_state_size]
	outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
	                                   initial_state=initial_state,
	                                   dtype=tf.float32)

多层：

	# create 2 LSTMCells
	rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
	
	# create a RNN cell composed sequentially of a number of RNNCells
	multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
	
	# 'outputs' is a tensor of shape [batch_size, max_time, 256]
	# 'state' is a N-tuple where N is the number of LSTMCells containing a
	# tf.contrib.rnn.LSTMStateTuple for each cell
	outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
	                                   inputs=data,
	                                   dtype=tf.float32)

