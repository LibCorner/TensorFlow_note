# 实现神经网络模型的基本步骤
--------------

一般神经网络和深度学习模型都是使用梯度下降的方法来训练，所以神经网络的本质就是通过神经网络模型得到输出，再根据教师信号（数据标签）得到一个目标函数，然后计算这个目标函数对模型参数的导数，并沿着梯度的负方向对模型中的参数进行更新。基本上所有的神经网络模型都是按照这样的步骤来实现的，他们的区别主要在两方面，一是数据从输入到输出在模型的内部计算过程不同，二是目标函数的计算过程有所不同。这两点也是实现自己的模型需要掌握的。

实现神经网络模型主要的难点其实就是在导数的计算和参数的更新上，还有就是从输入到输出的计算。而这些都可以使用Tensorflow或Theano来实现，这些工具有很多相似之处，它们都是通过构造符号计算图来构建模型，也都可以实现梯度的自动计算。这样就为我们省去了很多麻烦。而且tensorflow对一些常用的优化方法进行了封装，比如SGD,adadelta等。因此我们可以很容易实现一个深度学习的网络模型。

构建和实现神经网络模型的基本步骤有主要有三个：
×  构建从输入到输出的计算过程
×  根据模型的输出和标签数据计算损失函数，即目标函数
×  计算梯度并更新参数（训练）

#输入
tensorlow使用placeholder作为输入数据的占位符，训练时使用真实的数据替换这些占位符,在训练时使用`feed_dict`来替换这些占位符。placeholder函数的定义如下：
`tf.placeholder(dtype,shape=None,name=None)`
它有三个参数：
× dtype:输入tensor里的元素的类型。
* shape:输入tensor的shape,这是个可选参数，如果没有指定具体值，就可以输入任意shape的数据。
* name:该输入操作的名字。

## 计算推理过程

###  定义和初始化权重

1. 变量

×  变量用来维护计算图里的状态，比如网络的权重。计算图中变量都使用`tf.Variable`来定义，它是tensorflow里的一个类。
× `Variable()`的构造函数需要为变量提供一个初始值，这个初始值定义了该变量的type和shape。变量的值也可以通过`assign`方法来改变。
× 在用`run`方法前进行计算前，必须对变量进行初始化，可以通过run变量的`initializer`来初始化变量，也可以从文件中载入变量的值初始化，或运行一个`assign`操作为变量赋一个值。变量的`initializer`本质上也是一个`assgin`操作，它把变量的初始值赋值给该变量。
```
with tf.Session() as sess:
    #运行变量的initializer
    sess.run(w.initializer)
``
最常用的初始化方法是`initialize_all_variables()`,可以直接初始化所有的变量。
`sess.run(tf.initialize_all_variables())`

tensorflow_0.12使用`tf.global_variables_initialize()`代替`tf.initialize_all_variables()`

* 构造方法：`tf.Variable.__init__(initial_value,trainable=True,collections=None,validate_shape=True,name=None)`

2. 权重的初始化
权重的初始化一般使用tensorflow的常量（Constants）方法和随机（Random）方法来生成常量或随机数。

× `tf.truncated_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)`:

该方法根据给定的均值（mean）和标准差(stddev)，生成一个截断的正态随机分布,返回服从这个分布的二维tensor。

× `tf.zeros(shape,dtype=tf.float32,name=None)`:生成所有元素都是0的tensor.

* `tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)`

根据正态分布生成随机值。

× `tf.random_uniform(shape,minval=0.0,maxval=1.0,dtype=tf.float32,seed=None,name=None)`

根据均匀分布生成随机值。

### 计算
3. 矩阵运算

× 矩阵乘法
`tf.matmul(a,b,transpose_a=False,transpose_b=False,a_is_sparse=False,b_is_sparse=False,name=None)`
矩阵a乘矩阵b,即a*b。输入的必须是二维的矩阵，输入的两个矩阵的内部维数必须匹配，而且数据类型要一致。

* 批量矩阵乘法
`tf.batch_matmul(x,y,adj_x=None,adj_y=None,name=None)`

计算两个tensors的所有的切片的乘积。输入的tensor是3-D或更高维。

* 矩阵行列式
`tf.matrix_determinant(inupt,name=None)`

计算一个方阵的行列式

* 方阵求逆
`tf.matix_inverse(input,name=None)`

输入的必须是方阵

4. 基本数学运算

× 加、减、乘、除、取模
`tf.add(x,y,name=None)`
`tf.sub(x,y,name=None)`
`tf.mul(x,y,name=None)`
`tf.div(x,y,name=None)`
`tf.mod(x,y,name=None)`

这些函数的输入都是tensor，且x,y的类型应该一致。支持广播，每个操作都是element-wise进行的。

* 求绝对值
`tf.abs(x,name=None)`

* 求负数
`tf.neg(x,name=None)`

* sign函数
`tf.sign(x,name=None)`
逐项求元素的sign()值，大于0返回1,小于0返回-1，等于0,返回0.

× 求平方值
`tf.square(x,name=None)`
逐项计算元素的平方。

× 取整
`tf.round(x,name=None)`

* 平方根
`tf.sqrt(x,name=None)`

× 平方根的倒数
`tf.rsqrt(x,name=None)`

* x的y次方
tf.pow(x,y,name=None)

* expmZ
* log
`tf.log(x,name=None)`

* 两者取最大值
`tf.maximum(x,y,name=None)`
逐项比较x,y，返回最大的值，可以广播。
`tf.minimum(x,y)`取最小值

× 计算cosine值
`tf.cos(x,name=None)`
逐项计算x中元素的cos值。

5. Reduction: tensorflow提供了一些计算一般数学计算的方法，这些操作减少tensor的维度。

× `tf.reduce_sum(input_tensor,reduction_indices=None,keep_dims=False,name=None)`
该方法沿着某个维度计算元素的和。
`input_tensor`沿着给定的`reduction_indeces`的维度求和，维数也减少一维，除非`keep_dims`设为true。
如果`reduction_indices`没有给定，所有的维都会被reduced,并得到一个single 元素。

* `tf.reduce_prod(input_tensor,reduction_indeces=None,keep_dims=False,name=None)`
该方法沿着给定维度`reduction_indices`求元素的积。

× `tf.reduce_min(input_tensor,reduction_indeces=None,keep_dims=False,name=None)`
该方法计算`reduction_indices`维的最小值,

    最大值用`reduce_max`
    平均值用`reduce_mean`

6. 序列比较和索引

× `tf.argmin(input,dimension,name=None)`
返回tensor的`dimension`维的最小值的下标。

× `tf.argmax(input,dimension,name=None)`与argmin类似

× `tf.listdiff(x,y,name=none)`
该方法计算两个list的不同元素的个数，返回两个值：x中有而y中没有的元素值out和下标idx。

7. 激活函数

× `tf.nn.relu(features,name=None)`
计算修正线性：max(featrues,0),其中features是一个tensor.

* `tf.nn.relu6(features,name=None)`
计算 Rectified Linear 6:min(max(features,0),6)

* `tf.nn.softplus(features,name=None)`
计算softplus: log(exp(features)+1)

* `tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None)`
计算dropout

* `tf.nn.bias_add(value,bias,name=None)`
给value加一个bias

* `tf.sigmoid(x,name=None)`
计算sigmoid,y=1/(1+exp(-x))

* `tf.tanh(x,name=None)`
计算hyperbolic tangent of x

## 计算损失Loss








## 训练
