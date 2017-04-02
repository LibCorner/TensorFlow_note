#coding:utf-8
import tensorflow as tf

def max_pool1d(x,ksize=2):
    x=tf.expand_dims(x,axis=1)
    pool=tf.nn.max_pool(value=x,ksize=[1,1,ksize,1],strides=[1,1,ksize,1],padding="VALID",data_format='NHWC')
    s=tf.shape(pool)
    pool=tf.reshape(pool,[s[0],s[2],s[3]])
    return pool
def up_sample1d(x, rep,seq_len=1,axis=1):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Returns
        A tensor.
    """
    # slices along the repeat axis
    try:
        splits = tf.split(value=x, num_or_size_splits=seq_len, axis=axis)
    except TypeError:
        splits = tf.split(value=x, num_split=seq_len, split_dim=axis)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(x_rep, axis)