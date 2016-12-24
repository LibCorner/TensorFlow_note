# Tensorflow sequence2sequence 
tutorial:https://www.tensorflow.org/tutorials/seq2seq/

##  Seq2Seq模型
1.tf.nn.seq2seq.basic_rnn_seq2seq: 基本的RNN-RNN模型
```
def basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,dtype=dtypes.float32,scope=None)

这个方法先执行rnn来把encoder_inputs编码成state vector，然后运行由最后一个encoder state初始化的decoder。Encoder和Decoder使用相同的类型的RNN cell但不共享参数。
1. encoder: _,enc_state=rnn.rnn(cell,encoder_inputs,dtype=dtype)
2. decoder: tf.nn.seq2seq.rnn_decoder(decoder_inputs,enc_state,cell)
```


2.tf.nn.seq2seq.tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.共享rnn参数。
```
def tied_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,loop_function=None,dtype=dtypes.float32,scope=None)

先执行rnn把encode_inputs encode成state vector,再运行decoder,由最后一个encoder state初始化。
Encoder和decoder使用同一种类型的RNN cell并且共享参数。

with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
  #变量作用范围
  scope=scope or "tied_rnn_seq2seq"
  _,enc_state=rnn.rnn(cell,encoder_inputs,scope=scope)
  #reuse变量,使变量可以共享
  variable_scope.get_variable_scope().reuse_variables()
  return rnn_decoder(decoder_inputs,enc_state,cell,loop_function=loop_function,scope=scope) 
```

3.tf.nn.seq2seq.embedding_rnn_seq2seq: The basic model with input embedding.


4.tf.nn.seq2seq.embedding_tied_rnn_seq2seq: The tied model with input embedding.


5.tf.nn.seq2seq.embedding_attention_seq2seq: 加入了输入embedding和attention机制


## Decoder
1.tf.nn.seq2seq.rnn_decoder: seq2seq模型的decoder
```
def rnn_decoder(decoder_inputs,initial_state,cell,loop_function=None,scope=None)

参数：
  decoder_inputs: A list of 2D Tensors[batch_size,input_size]
  initial_state: 2D Tensor, shape=[batch_size,cell.state_size],用于初始化cell的state
  cell:rnn_cell.RNNCell 定义cell function 和size
  loop_function: 如果不是None,这个函数会应用到第i个output来生成第i+1个input,并且忽略掉decoder_inputs中除第一个以外的输入.loop_function(pre,i)=next
  scope:VariableScope for the create subgraph; defaults to 'rnn_decoder'.

Returns:
  (outputs,state)形式的元组
```

2.tf.nn.seq2seq.embedding_rnn_decoder: 加入了embding和pure-decoding选项
```

```
