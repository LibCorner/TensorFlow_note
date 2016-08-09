#tensorflow的卷积操作

##  二维卷积
`tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)`

1. 参数
* 输入input是一个4-D的tensor,它的shape是(batch_size,in_height,in_width,in_channels)_
× filter是卷积核，即网络中的权重，shape为(filter_height,filter_width,in_channels,out_channels)_
×  strides:一个int列表，长度为4,表示滑动窗口在input每一维的步长。一般strides=(1,stride,stride,1) 
* padding: `SAME`或`VALID`,选择不同的padding方式。
× use_cudnn_on_gpu: bool类型。
× name: 该操作的名称。
_
2. 卷积操作的步骤：
× 把filter转换成2-D的矩阵(filter_height*filter_width*in_channels,output_channels)
* 从输入tensor中抽取image patches,形成虚拟tensor，shape为(batch,out_height,out_width,filter_height*filter_with*in_channels)
* 对于每个patch, filter矩阵和image patch向量相乘。
_


