#coding:utf-8
import tensorflow as tf

'''
1. 对角矩阵
1.1 tf.diag(diagonal,name=None)
参数：
    * diagonal: Tensor, Rank最大为3, 对角矩阵的每个对角元素的值
    * name
返回：
    返回一个对角tensor，对角的值为给定的diagnoal values
    
1.2 tf.diag_part(input,name=None)
返回输入tensor的对角矩阵的对角元素。
参数：
    * input: Tensor, Rank k tensor, k is 2,4,or 6
返回：
    * A Tensor.输入矩阵的对角元素的值。与tf.diag过程相反。
    
1.3 tf.matrix_diag(diagnoal,name=None)
给定一个batch的对角元素值，返回一个batch的对角矩阵。
参数：
    * diagonal: Tensor, Rank k, k>=1
返回：
    * Tensor, Rank k+1,shape=[diagonal.shape]+[diagonal.shape[-1]]
    
1.4 tf.matrix_diag_part(input,name=None)
与tf.matrix_diag过程相反，返回batch个tensor的对角元素。

1.5 tf.matrix_set_diag(input,diagonal,name=None)
使用diagonal的值替换(设置)input的主对角线的元素,其他元素不变。

1.6 tf.eye(num_rows,num_columns=None,batch_shape=None,dtype=tf.float32,name=None)
构造单位矩阵。
'''
values=tf.constant([2,3,4,5])
diag=tf.diag(values,name="diag")
sess=tf.Session()
print(sess.run(diag))

#diag_part
part=tf.diag_part(diag)
print(sess.run(part))
print()

#matrix_diag
matrix_diag=tf.matrix_diag(diagonal=[[1,2,3,4],[5,6,7,8]])
print(sess.run(matrix_diag))
#matrix_diag_part
part=tf.matrix_diag_part(matrix_diag)
print(sess.run(part))
#matrix_set_diag
new_diag=tf.matrix_set_diag(matrix_diag,[[3,3,3,4],[4,5,4,5]])
print(sess.run(new_diag))


'''
2. 转置
2.1 tf.transpose(a,perm=None,name='transpose')
根据参数`perm`重排Tensor的维度

2.2 tf.matrix_transpose(a,name=)
对a的最后两个维度进行转置，rank(a)>=2

2.3 tf.sparse_transpose(sp_input,perm=None,name=None)
对`SparseTensor`进行转置, `SparseTensor`数据存储形式为`indices: values`
参数：
    * sp_input: `SparseTensor`
    * perm: 维度的排列
'''

'''
3. square矩阵求逆矩阵: tf.matrix_inverse(input,adjoin=None,name=None)
    * 求一个或多个矩阵的逆矩阵或伴随矩阵(adjoints=True).
    * 使用LU分解计算逆
'''
'''
4. 矩阵乘法： 
4.1 tf.matmul(a,b,transpose_a=False,transpose_b=False,adjoint_a=False,adjoint_b=False,
              a_is_sparse=False,b_is_sparse=False,name=None)
参数：
    * 
4.2 tf.sparse_matmul(a,b,transpose_a=None,transpose_b=None,a_is_sparse=None,
                     b_is_sparse=None,name=None)

'''

'''
'''