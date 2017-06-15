#coding:utf-8
import tensorflow as tf

'''
1.加减乘除等算术运算
* tf.add :加
* tf.substract: 减法
* tf.multiply: 乘法
* tf.div: 除法，服从python2的除法规则,x/y，如果x,y都是整数结果为int类型，否则结果为float
* tf.divide:除法
* tf.truediv: 使用python3的除法操作语义，x/y, 结果为float，如果要使用整除，使用x//y或tf.floordiv
* tf.floordiv: 除法，取整
* tf.truncatediv
* tf.mod: 取模（取余）
'''
x=tf.constant(7)
y=tf.constant(4)
mul=tf.multiply(x,y)
div=tf.div(x,y)
divide=tf.divide(x,y)
truediv=tf.truediv(x,y)
truncatediv=tf.truncatediv(x,y)
sess=tf.Session()
out=sess.run([mul,div,divide,truediv,truncatediv])
print(out)

'''
2. pairwise 交叉相乘
* tf.cross(a,b,name): 计算a,b中元素的两两之间的乘积
'''
a=tf.constant([1,2,3])
b=tf.constant([0,2,3])
cross=tf.cross(a,b)
out=sess.run(cross)
print(out)


'''
3. 累加
tf.add_n(inputs,name=None)
计算inputs中所有tensor的和
参数：
    inputs: a list of Tensors
'''
add_n=tf.add_n([a,b])
out=sess.run(add_n)
print(out)

'''
4. 绝对值，符号函数,倒数，取整
* tf.abs: 绝对值
* tf.sign: x的符号，-1,0,1（负，0，正）
* tf.negative(x,name): 取负值，y=-x
* tf.reciprocal(x,name=None): 计算倒数，y=1/x
* tf.round(x,name=None): 四舍五入
*  tf.ceil(x): 向上取整
* tf.floor: 向下取整
'''

'''
5. 指数函数，对数函数
* tf.sqrt: 开根号
* tf.pow(x,y):x的y次方，x^y
* tf.rsqrt(x):开根号的倒数
* tf.exp: e^x
* tf.log: 对数
'''

'''
6.比较
* tf.maximum
* tf.minimum
'''

'''
7. 三角函数
* tf.cos
* tf.sin
* tf.tan
'''