#coding:utf-8
import tensorflow as tf
import numpy as np

'''
beam search 解码
'''

'''
1. tf.nn.ctc_beam_search_decoder(inputs,sequence_lenth,beam_width=100,top_paths=1,merge_repeated=True)
* 在输入的logits上执行beam search.
* `ctc_greedy_decoder`是`ctc_beam_search_decoder`的`top_paths=1`和`beam_width=1`特殊情况（但是比这种特殊情况要快）
参数：
    * inputs: 3-D float Tensor logits, shape=[max_time, batch_size, num_classes].
    * sequence_length: 1-D int32向量包含序列的长度， shape=[batch_size]
    * beam_width: int scalar>=0（beam search width）
    * top_paths: int scalar>=0, <=beam_width（控制输入的大小）
    * merge_repeated: Boolean,默认为True。如果为True, 合并输出中相同的classes,比如序列`A B B * B * B`（*为blank label）会变成`A B`, 如果为False输出为:`A B B B B B`。
返回：
    * 元组(decoded,log_probabilities)
    * decoded是一个single-element list。 decoded[0]是一个`SparseTensor`包含decoded outputs:
        (1)decoded.indices:indices matrix（total_decoded_outputs*2）, The rows store：[batch,time]
        (2)decoded.values: Values vector,size(total_decoded_outputs). The vector存储decoded classes
        (3)decocded.shape: Shape vector, size(2),[batch_size,max_decoded_length]
'''

'''
2. tf.nn.ctc_greedy_decoder(inputs,sequence_length,merge_repeated=True)
在输入inputs上执行贪心算法的decoding（best path）.
参数：
    * inputs: 3-D float Tensor logits, shape=[max_time, batch_size, num_classes].
    * sequence_length: 1-D int32向量包含序列的长度， shape=[batch_size]
    * merge_repeated: Boolean,默认为True。如果为True, 合并输出中相同的classes,比如序列`A B B * B * B`（*为blank label）会变成`A B`, 如果为False输出为:`A B B B B B`。
返回：
    * 元组(decoded,log_probabilities)
    * decoded是一个single-element list。 decoded[0]是一个`SparseTensor`包含decoded outputs:
        (1)decoded.indices:indices matrix（total_decoded_outputs*2）, The rows store：[batch,time]
        (2)decoded.values: Values vector,size(total_decoded_outputs). The vector存储decoded classes
        (3)decocded.shape: Shape vector, size(2),[batch_size,max_decoded_length]
'''

batch_size=20
logits=np.random.rand(10,batch_size,50).astype(np.float32)
sequence_length=np.ones(shape=[batch_size,])
sequence_length[:]=5
inputs=tf.nn.softmax(logits)
#decoded=tf.nn.ctc_greedy_decoder(inputs=inputs,sequence_length=sequence_length)
decoded=tf.nn.ctc_beam_search_decoder(inputs=inputs,sequence_length=sequence_length,beam_width=5,top_paths=5,merge_repeated=False)
sess=tf.Session()
outputs,log_p=sess.run(decoded)