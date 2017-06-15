#coding:utf-8
import tensorflow as tf
#tf.reset_default_graph()
'''
with tf.device("/cpu:0"):
    
    a=tf.Variable(tf.zeros([1000,1024]))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(a)
'''

#with tf.device("/cpu:0"):
#    b=tf.Variable(tf.ones([2000,512]))
#    tf.assign(b,b+1)
#    sess=tf.Session()
#    sess.run(tf.global_variables_initializer())
#    saver=tf.train.Saver()
#    saver.restore(sess,"/tmp/w.ckpt")
#    out=sess.run(b)
#    print(out)
#    saver.save(sess,"/tmp/w.ckpt")
    
#with tf.Graph().as_default():
with tf.device("/cpu:0"):
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