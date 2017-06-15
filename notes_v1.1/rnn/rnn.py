#coding:utf-8
import tensorflow as tf

'''
1. tf.contrib.rnn.static_rnn: 使用特定的RNNCell创建一个循环神经网络
(1) 最简单的RNN网络生成过程为：
    state=cell.zero_state(...)
    outputs=[]
    for input_ in inputs:
        output,state=cell(input_,state)
        outputs.append(output)
    return (outputs,state)
(2) 动态的RNN网络: 使用`sequence_length`参数
    * 如果提供了`sequence_length` vector, 会执行动态的计算， 
    * 不计算minibatch中超过最大长度的部分（从而节省了计算时间）
    * 并且能够正确的传播state到最后的final state output.
    * 在时刻t, batch row为b的 动态计算过程为:
        (output,state)(b,t)=
            (t>=sequence_length(b))
                ? (zeros(cell,output_size),states(b,sequence_length(b)-1))
                :cell(input(b,t),state(b,t-1))

(3)参数：
    * cell: RNNCell实例
    * inputs: 长度为T的inputs list,每个inputs的shape为[batch_size,input_size]
    * initial_state: 可选。 If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size]. If cell.state_size is a tuple, this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size.
    * dtype:可选。
    * sequence_length: 类型为int32的tensor, shape为[batch_size], values in[0,T)
    * scope: 
(4)返回：
    outputs,state    
'''

'''
2. tf.contrib.rnn.static_bidirectional_rnn: 双向rnn
(1) 简介：
    * 与单向的情况类似，但是使用两个独立的forward和backward RNNs, 
    * 最后得到的forward outputs和backward output使用concatenate结合，shape为[time,batch,cell_fw.output_size+cell_bw.output_size]
    * forward和backward cell的input_size必须一致。
    * 两个方向的初始state为zero（但是可以选择）
    *  no intermediate states are ever returned -- the network is fully unrolled for the given (passed in) length(s) of the sequence(s) or completely unrolled if length(s) is not given.
(2)参数：
    * cell_fw: RNNCell实例
    * cell_bw: RNNCell实例
    * inputs: 长度为T的inputs的list, 每个input的shape为[batch_size,input_size]
    * initial_state_fw:可选， This must be a tensor of appropriate type and shape [batch_size, cell_fw.state_size]. If cell_fw.state_size is a tuple, this should be a tuple of tensors having shapes [batch_size, s] for s in cell_fw.state_size.
    * initial_state_bw: 可选
    * dtype: 
    * sequence_length: shape为[batch_size],类型为int32/int64
    * scope
(3)返回
    (outputs,output_state_fw,output_state_bw)
'''

'''
3. tf.contrib.rnn.stack_bidirectional_rnn: 堆叠多个双向rnn层。
'''
