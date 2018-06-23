### 条件语句： tf.cond

	cond(
		pred,
		true_fn=None,
		false_fn=None,
		strict=False,
		name=None,
		fn1=None,
		fn2=None
	)

如果`pred`是True, 则返回`true_fn()`, 否则返回`false_fn()`

### 切片： tf.slice

	slice(
		input_,
		begin,
		size,
		name=None
	)
> * 从输入的tensor`input`中抽取大小为`size`的切片
> * `begin`表示每个维度切片的开始位置, 从0开始
> * `size` 表示每个维度切片的元素个数, 从 1开始
> > 如果`size[i]=-1`, 第`i`维剩下的所有的元素都在slice中，换言之，它等同于:
> > ```size[i]=input.dim_size(i)-begin[i]```

比如：

	t = tf.constant([[[1, 1, 1], [2, 2, 2]],
	                 [[3, 3, 3], [4, 4, 4]],
	                 [[5, 5, 5], [6, 6, 6]]])
	tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
	tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
	                                   #   [4, 4, 4]]]
	tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
	                                   #  [[5, 5, 5]]]

**注意： `tf.Tensor.__getitem__`可以用python的方式来切片，它允许使用`foo[3:7,:-2]` 来代替`tf.slice([3,0],[4,foo.get_shape()[1]-2])`**