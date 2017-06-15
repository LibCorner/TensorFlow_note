#coding:utf-8
import tensorflow as tf
import numpy as np
from pprint import pprint
'''
seq2seq： Attention
tf.contrib.seq2seq
'''

'''
1. Attention wrapper和Attention Mechanism: 
    （1）包装了其他RNNCell的RNNCell对象，并且实现attention机制。
    （2）attention的形式由`tf.contrib.seq2seq.AttentionMechanism`的子类决定，这些子类描述了创建wrapper时使用的attention的形式(additive或multiplicative).
    （3）`AttentionMechanism`实例由一个`memory` tensor参数构造，这个`memory`tensor可以创建keys和values tensor。
    （4）两种基本的attention机制为：
        * `tf.contrib.seq2seq.BahdanauAttention`: additive attention
        * `tf.contrib.seq2seq.LuongAttention`: multiplicatie attention
    (5) 传递给AttentionMechanism构造器的`memory` tensor的shape为[batch_size,memory_max_time,memory_depth]
    (6) attention机制有depth的概念， depth由参数num_units决定(depth其实就是向量的维度):
        * 对于像BahdanauAttention这样的attention, queries和memory都被映射成depth为`num_units`的tensors;
        * 而对与其他的attention，比如LuongAttention, `num_units`要与queries的depth一致，memory tensor要映射成这个depth的tensor。
2. Attention Wrappers
基本的AttentionWarpper: tf.contrib.seq2seq.DynamicAttentionWrapper
（1）接收一个RNNCell实例，一个AttentionMechanism实例和一个attention depth参数，以及一些其他可选参数。
（2）每个时刻的基本计算过程为：
    cell_inputs = concat([inputs, prev_state.attention], -1)  #inputs与attention结合
    cell_output, next_cell_state = cell(cell_inputs, prev_state.cell_state) #rnncell的输出
    score = attention_mechanism(cell_output) #计算attention得分
    alignments = softmax(score)
    context = matmul(alignments, attention_mechanism.values)
    attention = tf.layers.Dense(attention_size)(concat([cell_output, context], 1))
    next_state = DynamicAttentionWrapperState(
      cell_state=next_cell_state,
      attention=attention)
    output = attention
    return output, next_state
（3）一些中间计算是可以配置的，比如：
    * `inputs`和`prev_state.attention`的initial concatenation可以替换成其他的mixing function。
    * `softmax`函数可以替换成其他的可选项
    * 最后，outputs可以配置成`cell_output`而不是`attention`
(4) 使用`DynamicAttentionWrapper`的好处是可以很好的与其他的wrapper结合使用。
(5). tf.contrib.seq2seq.DynamicAttentionWrapper
属性：
    * output_size
    * state_size
方法：
    (1)__init__(cell,attention_mechanism,attention_size,cell_input_fn=None,probability_fn=None,output_attention=True,name=None)
    参数：
        * cell: RNNCell实例
        * attention_mechanism: `AttentionMechanism`实例。
        * attention_size: integer, attention(output) tensor的depth
        * cell_input_fn: 可选，可调用的函数，默认为: lambda inputs,attention: array_ops.concat([inputs,attention],-1)
        * probability_fn: 可选，可调用的函数，把attention score转换成概率形式, 
                            默认使用tf.nn.softmax, 
                            其他选择包括： tf.contrib.seq2seq.hardmax和tf.contrib.sparsemax.sparsemax
        * output_attention: boolean, 如果True(默认), 输入attention值(Luong-style)，否则输出`cell`的output（Bhadanau-style）。
                            两种attention都会把attention tensor通过state传播到下一个time step。
        * name:
    (2)__call__(inputs,state,scope=None):
        attention-wrapped RNN的执行过程：
            Step 1: Mix the inputs and previous step's attention output via cell_input_fn.
            Step 2: Call the wrapped cell with this input and its previous state.
            Step 3: Score the cell's output with attention_mechanism.
            Step 4: Calculate the alignments by passing the score through the normalizer.
            Step 5: Calculate the context vector as the inner product between the alignments and the attention_mechanism's values (memory).
            Step 6: Calculate the attention output by concatenating the cell output and context through the attention layer.
        参数：
            * inputs: 每个时刻的输入
            * state: `DynamicAttentionWrapperState`实例
            * scope: 必须为None.
        返回:
            (attention,next_state)
    (3)zero_state(batch_size,dtype)    
'''

'''
3. tf.contrib.seq2seq.BahdanauAttention
属性：
    * keys
    * memory_layer
    * query_layer
    * values
参数：
    * num_unites: attention机制的depth
    * memory: The memory to query, 通常是RNN encoder的输出，shape为[batch_size,max_time,...]
    * memory_sequence_length(可选): memory的序列的长度。
    * normalize: boolean, 是否normalize the energy term.
    * attention_r_initializer: 默认为0,  Initial value of the post-normalization bias when normalizing.
    * name: name.
方法:
    (1)__call__(query):计算attention  score.
        * query: 与self.values类型匹配的Tensor, shape为[batch_size,query_depth]
        * score: tensor, shape为[batch_size,max_time], max_time为memory的max_time.
4. tf.contrib.seq2seq.LuongAttentioin
'''
batch_size=16
max_time=10
input_dim=5
data=np.random.rand(batch_size,max_time,input_dim)
query_data=np.random.rand(batch_size,input_dim)
tf.reset_default_graph()
memory=tf.placeholder(dtype=tf.float32,shape=[batch_size,max_time,input_dim])
query=tf.placeholder(dtype=tf.float32,shape=[batch_size,input_dim])
#使用Attention机制计算attention score
att=tf.contrib.seq2seq.BahdanauAttention(num_units=3,memory=memory,memory_sequence_length=None)
scores=att(query)
    
sess=tf.Session()
sess.run(tf.global_variables_initializer())
out=sess.run(scores,feed_dict={memory:data,query:query_data})
print(out)
pprint(["可训练的参数：",tf.trainable_variables()])
sess.close()


