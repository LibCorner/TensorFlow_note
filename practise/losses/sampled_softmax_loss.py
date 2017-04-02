#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

num_samples=10
target_vocab_size=1000
size=100


if num_samples>0 and num_samples<target_vocab_size:
    w=tf.Variable(tf.random_uniform([size,target_vocab_size],-1,1))
    #转置
    w_t=tf.transpose(w)
    b=tf.Variable(tf.zeros([target_vocab_size]))


inputs=tf.placeholder(tf.float64,shape=[None,size])
labels=tf.placeholder(tf.int64,shape=[None,1])
lables=tf.reshape(labels,[-1,1])
#参数类型转换成tf.float32
local_w_t=tf.cast(w_t,tf.float32)
local_b=tf.cast(b,tf.float32)
local_inputs=tf.cast(inputs,tf.float32)

"""计算和返回sampled softmax training loss.
1. 在类别数很多时可以用来快速训练softmax分类器的一种方式


2. 该操作只在训练的时候用，It is generally an underestimate of the full softmax loss.
   在推理的时候，可以使用full softmax概率，公式为： 
       `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

  See our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)
  Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
  ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.
"""
"""
  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

"""
#sampled_softmax_loss
loss=tf.nn.sampled_softmax_loss(local_w_t,local_b,local_inputs,labels,num_samples,target_vocab_size)
data=np.random.rand(10,100)
label_data=np.random.randint(low=0,high=target_vocab_size,size=(10,1))
with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    init=tf.global_variables_initializer()
    init.run()
    print(sess.run(loss,feed_dict={inputs:data,labels:label_data}))
