# -*- coding: utf-8 -*-
#word2vec_basic
import tensorflow as tf
import numpy as np
import math
import data

#建立词典并把不频繁的词替换成UNK
vocabulary_size=data.vocabulary_size
print(vocabulary_size)
#创建skip-gram模型

batch_size=128
embedding_size=128  #embedding向量的维度
skip_window=1  #考虑左右相邻词的个数
num_skips=2   #how many times to reuse an input to gernerate a label

#随机选取验证集
valid_size=16  #验证集的数量
valid_window=100  #只在分布排名靠前的词中选取
valid_examples=np.random.choice(valid_window,valid_size,replace=False)
num_sampled=64  #采样负样本的个数

graph=tf.Graph()

with graph.as_default():
    
    #Input data
    train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
    train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    
    #使用CPU
    with tf.device('/cpu:0'):
        #look up embedding for inputs
        embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
        embed=tf.nn.embedding_lookup(embeddings,train_inputs)
        
        #构造NCE loss的权重
        nce_weights=tf.Variable(
                                tf.truncated_normal([vocabulary_size,embedding_size],
                                                    stddev=1.0/math.sqrt(embedding_size)))
        nce_biases=tf.Variable(tf.zeros([vocabulary_size]))
        
        #计算batch的平均NCE loss
        #tf.nce_loss每次自动采样新的negtive labels
        loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=embed,
                            num_sampled=num_sampled,
                            num_classes=vocabulary_size))
        
        #构造SGD optimizer
        optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
        #计算minibatch examples和所有embeedings之间的相似度cosine相似度
        norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embedding=embeddings/norm
        valid_embeddings=tf.nn.embedding_lookup(normalized_embedding,valid_dataset)
        similarity=tf.matmul(valid_embeddings,normalized_embedding,transpose_b=True)
        
        #tensorflow变量初始化
        init=tf.global_variables_initializer()
        
#训练
num_steps=1001

with tf.Session(graph=graph) as session:
    #初始化所有变量
    init.run()
    print("Initialized")
    
    average_loss=0
    for step in range(num_steps):
        batch_inputs,batch_labels=data.generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
        
        #更新参数
        _,loss_val=session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss+=loss_val
        
        if step%200==0:
            if step>0:
                average_loss/=2000
            print("Average loss at step",step,":",average_loss)
            average_loss=0
            
        #计算验证样本的相似度
        if step %1000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=data.reverse_dictionary[valid_examples[i]]
                top_k=8 #最相似的词的个数
                nearest=(-sim[i,:]).argsort()[1:top_k+1]
                log_str="Nearest to %s" %valid_word
                for k in range(top_k):
                    close_word=data.reverse_dictionary[nearest[k]]
                    log_str="%s %s," % (log_str,close_word)
                print(log_str)
                
        final_embeddings=normalized_embedding.eval()
        
import matplotlib.pyplot as plt
#可视化词向量
def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0]>=len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        plt.savefig(filename)
        

#降维可视化
try:
    from sklearn.manifold import TSNE
    
    tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_only=500
    #降维
    low_dim_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
    labels=[data.reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs,labels)
    
except ImportError:
    print("Please install sklearn,matplotlib,and scipy to visualize embedding.")
    

    
        
