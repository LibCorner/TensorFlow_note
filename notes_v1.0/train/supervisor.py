#coding:utf-8
import tensorflow as tf
'''
tf.train.Supervisor:
    * A training helper: checkpoint models（抽点检验模型）和computes summaries
    * Supervisor是一个包装器，包装了`Coordinator`,`Saver`,`SessionManager`类   
'''

'''
1. 单个程序中使用
* 使用`with sv.managed_session()`代码块
* 如果一个程序崩溃(crashes)了，在重新启动时managed session自动从最近一次的checkpoint重新初始化变量。
* 循环训练时要检查`shold_stop()`,Supervisor接收任何exception的通知，如果raise一个exception，`should_stop()`就会返回True.
'''
with tf.Graph().as_default():
    a=tf.Variable(0)
    op=tf.assign(a,a+1)
    sv=tf.train.Supervisor(logdir='tmp/mydir')
    i=0
    '''每次运行会自动加载最近一次保存的checkpoint
    '''
    with sv.managed_session() as sess:
        #sess.run(tf.global_variables_initializer())
        while not sv.should_stop():
            _,o=sess.run([op,a])
            print(o)
            i+=1
            if i==10:
                break
print("====================")            
#with tf.Graph().as_default():
tf.reset_default_graph()
a=tf.Variable(0)
op=tf.assign(a,a+1)
sv=tf.train.Supervisor(logdir='tmp/mydir')
i=0
'''每次运行会自动加载最近一次保存的checkpoint
'''
with sv.managed_session() as sess:
    while not sv.should_stop():
        _,o=sess.run([op,a])
        print(o)
        i+=1
        if i==10:
            break
            
'''
2. Use for multiple replicas（复制品，复本）
2.1 简介
    * 当使用replicas训练的时候，需要在`Cluster`中部署相同的程序，
    * 这些task中的一个作为chief: 处理initialization, checkpoint,summaries和recovery
    * 其他的task依赖chief提供的的这些services.
    * 只需改变single program的代码，在Superviser中指定该program是否是chief
    * 选择chief, 可以根据server_def.task_index, job_def.name或job_def.tasks.
    * 只能有一个chief
2.2 执行过程
    * 在chief task中，`Supervisor`与single program相同。
    * 在其他task中，`sv.managed_session()`等待模型初始化，然后返回session。
    * 非chief的task依赖chief task初始化模型。
    * 如果有一个task崩溃并重启，managed_session()会检查模型是否已经初始化，如果是，直接创建一个session, 否则要先由chief task初始化。
'''
is_chief=True

with tf.Graph().as_default():
    a=tf.Variable(0)
    op=tf.assign(a,a+1)
    sv=tf.train.Supervisor(logdir='tmp/mydir',is_chief=is_chief)
    i=0
    '''每次运行会自动加载最近一次保存的checkpoint
    '''
    with sv.managed_session() as sess:
        while not sv.should_stop():
            _,o=sess.run([op,a])
            print(o)
            i+=1
            if i==10:
                break