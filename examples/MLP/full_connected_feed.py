#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

learning_rate=0.01
max_steps=2000
hidden1_unit=128
hidden2_unit=32
batch_size=100
train_dir='mnist-data'
fake_data=False


import mnist
import mnist_data as input_data
import time

def placeholder_inputs(batch_size):
    '''创建占位符'''
    image_placeholder = tf.placeholder(tf.float32, shape=[batch_size,mnist.IMAGE_PIXELS])
    label_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return image_placeholder,label_placeholder

def fill_placeholder(data_set,image_pl,labels_pl):
    """返回tensorflow的feed字典"""
    image_feed,label_feed=data_set.next_batch(batch_size)
    feed_dict={image_pl:image_feed,labels_pl:label_feed}
    return feed_dict

def do_eval(sess,eval_correct,image_placeholder,label_placeholder,dataset):
    '''计算准确率'''
    true_count=0
    steps_per_epoch = dataset.num_examples // batch_size
    num_examples=steps_per_epoch * batch_size #总数

    for step in xrange(steps_per_epoch):
        feed_dict=fill_placeholder(dataset,image_placeholder,label_placeholder)
        true_count+=sess.run(eval_correct,feed_dict=feed_dict)

    precision=1.0 * true_count / num_examples
    print('Number of examples: %d Num correct: %d Precision: @ 1: %.4f' %(num_examples,true_count,precision))

def run_training():
    data_sets=input_data.read_data_sets(fake_data)
    
    with tf.Graph().as_default():
        image_placeholder,label_placeholder = placeholder_inputs(batch_size)

        logits=mnist.inference(image_placeholder,hidden1_unit,hidden2_unit)
        #计算损失
        loss=mnist.loss(logits,label_placeholder)
        #训练
        train_op=mnist.training(loss,0.01)
        #计算正确分类数
        eval_correct=mnist.evaluation(logits,label_placeholder)
        #合并默认图中的所有summaries操作
        summary_op=tf.merge_all_summaries()
        #用于保存网络中的变量
        saver=tf.train.Saver()

        sess=tf.Session()
        #初始化所有变量
        sess.run(tf.initialize_all_variables())

        summary_writter=tf.train.SummaryWriter(train_dir,sess.graph)

        for step in xrange(max_steps):
            start_time=time.time()
            #获取feed_dict字典
            feed_dict=fill_placeholder(data_sets.train,image_placeholder,label_placeholder)

            _,loss_value=sess.run([train_op,loss],feed_dict=feed_dict)

            duration = time.time()-start_time

            if step % 100==0:
                print('Step %d:loss=%.2f (%.3f sec) %(step,loss_value,duration)')
                summary_str=sess.run(summary_op,feed_dict=feed_dict)
                summary_writter.add_summary(summary_str,step)
                summary_writter.flush()

            if step % 1000==0:
                saver.save(sess,train_dir,global_step=step)
                
                print('Training Data Eval:')
                do_eval(sess,eval_correct,image_placeholder,label_placeholder,data_sets.train)

                print("Validation Data Eval")
                do_eval(sess,eval_correct,image_placeholder,label_placeholder,data_sets.validation)

                print("Test Data Eval:")
                do_eval(sess,eval_correct,image_placeholder,label_placeholder,data_sets.test)



if __name__=="__main__":
    print("runing...")
    run_training()