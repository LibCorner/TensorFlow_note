#coding:utf-8
import tensorflow as tf
'''
负采样损失函数
1. tf.nn.nce_loss
注意：
    1. 默认使用log-unifrom(Zipfian)分布来采样，所以labels必须按照频率降序排列，以获得更好的结果。
    2. 当num_true>1时， 每个target class的概率为1/num_true。
参数：
    1. weights: [num_class,dim],decoder的embedding, 用于进行负采样并计算概率
    2. bias: [num_class],decoder的bias, 用于进行负采样并计算概率
    3. labels: [batch_size,num_true], 正样本的index
    4. inputs: [batch_size,dim],t时刻的输出向量
    5. num_sampled: 采样样本数量
    6. num_classes: 可能的类别数
    7. num_true: int, 每个训练样本的target class的个数。
    8. sampled_values: 通过一个采样函数返回的tuple:(sampled_candidates,true_expected_count, sampled_expected_count)，
                    如果是None, 默认使用log_uniform_candiate_sampler。
    9. remove_accidental_hits: bool, 是否移除`accidental hits`。
    10. partition_strategy: 目前支持`div`和`mod`, A string specifying the partitioning strategy。
'''

'''
2. tf.nn.sampled_softmax_loss

参数：
    1. weights: [num_class,dim],decoder的embedding, 用于进行负采样并计算概率
    2. bias: [num_class],decoder的bias, 用于进行负采样并计算概率
    3. labels: [batch_size,num_true], 正样本的index
    4. inputs: [batch_size,dim],t时刻的输出向量
    5. num_sampled: 采样样本数量
    6. num_classes: 类别总数，即输出词库的大小。
采样softmax损失函数的计算过程：
    1. 随机生成负样本的index
    2. 使用embedding_lookup操作得到对应的decoder embedding的sampled_weights和sampled_bias：shape分别为[sampled_num,dim],[sampled_num]
    3.使用矩阵乘法，计算每个采样样本的概率：matmul(inputs,sampled_weights)+sampled_bias, 得到shape为[batch_size,num_classes]
    4. 同样，计算正样本的概率
    5. 拼接正样本的概率和负样本的概率，并构造相应的labels
    6. 计算交叉熵损失
'''

