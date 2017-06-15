#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np
from pprint import pprint

class Seq2seqModel(object):
    def __init__(self,lr=0.001,l2=0.001):
        self.lr=lr
        self.l2=l2
        self.emb_dim=50
        self.hidden_dim=50
        self.dtype=tf.float32
        self.subject_len=5
        self.question_len=20
        self.vocab_size=1000
        self.embedding=np.random.rand(self.vocab_size,self.emb_dim)
        
        tf.reset_default_graph()
        
        self.question_in=tf.placeholder(dtype=tf.int32,shape=[None,self.question_len])
        self.subject_in=tf.placeholder(dtype=tf.int32,shape=[None,self.subject_len])
        self.seq_len=tf.placeholder(dtype=tf.int32,shape=[None,])
        
        self.init_weight()
        self.sess=tf.Session()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
    def init_weight(self):
        self.embedding=tf.Variable(self.embedding,dtype=self.dtype)
        self.dec_embedding=tf.Variable(self.embedding,dtype=self.dtype)
        self.dec_bias=tf.Variable(tf.zeros([self.vocab_size]))
        self.encoder_cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,input_size=None,state_is_tuple=True, activation=tf.nn.tanh)
        self.decoder_cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,input_size=None,state_is_tuple=True,activation=tf.nn.tanh)
        
    def build_model(self):
        questions=tf.nn.embedding_lookup(self.embedding,self.question_in)
        encoder_outputs,encoder_state=self.encoder(questions)
        print("encoder_outputs",encoder_outputs.get_shape())
        decoder_inputs=tf.zeros(shape=[tf.shape(encoder_outputs)[0],self.subject_len,self.hidden_dim])
        decoder_outputs,decoder_state,decoder_context=self.decoder(encoder_state=encoder_state,inputs=decoder_inputs)
        self.decoder_outptus=decoder_inputs
        print("decoder_outputs:",decoder_outputs.get_shape())
        self.logits,_,_=self.decoder(encoder_state,inputs=encoder_outputs,is_train=False)
        #beam search解码
        self.output=tf.nn.ctc_greedy_decoder(tf.transpose(self.logits,perm=[1,0,2]),sequence_length=self.seq_len,merge_repeated=False)
        weights=tf.cast(tf.greater(self.subject_in,0),tf.float32)
        
        def softmax_loss_function(inputs,labels):
            '''定义softmax_loss_function损失函数，由seq2seq.sequence_loss调用
            '''
            print("inputs:",inputs.get_shape())
            print("labels:",labels.get_shape())
            labels=tf.expand_dims(labels,1) 
            loss=tf.nn.sampled_softmax_loss(self.dec_embedding,self.dec_bias,labels=labels,inputs=inputs,num_sampled=100,num_classes=self.vocab_size)
            return loss 
        '''
        损失函数：
        logits: 3-D, shape=[batch_size,sequence_length,num_decoder_symbols], num_decoder_symbols为decoder的维度，可以是类别个数，也可以是词向量的维度（需要使用负采样的softmax_loss_function）
        targets: 2-D, shape=[batch_size,sequence_length], dtype为int,目标序列的index
        weights: 2-D, shape=[batch_size,sequence_length],dtype为float
        softmax_loss_function: Function(inputs-batch,labels-batch)
        '''
        self.loss=seq2seq.sequence_loss(logits=decoder_outputs,targets=self.subject_in,weights=weights,softmax_loss_function=softmax_loss_function)
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
    def encoder(self,inputs):
        '''编码器
        '''
        with tf.variable_scope("encoder") as scope:
            initial_state=None#self.encoder_cell.zero_state()
            try:
                outputs,state=tf.nn.dynamic_rnn(self.encoder_cell,inputs,initial_state=initial_state,
                                                dtype=self.dtype,time_major=False,scope=scope)
            except:
                self.sess.run(tf.global_variables_initializer())
                outputs,state=tf.nn.dynamic_rnn(self.encoder_cell,inputs,initial_state=initial_state,
                                                dtype=self.dtype,time_major=False,scope=scope)
        return outputs,state
    def decoder(self,encoder_state,inputs=None,is_train=True):
        '''
        解码器
        '''
        with tf.variable_scope("decoder") as scope:
            if is_train is True:
                decoder_fn=seq2seq.simple_decoder_fn_train(encoder_state)
                outputs,final_state,final_context_state=seq2seq.dynamic_rnn_decoder(self.decoder_cell,decoder_fn=decoder_fn,inputs=inputs,sequence_length=self.seq_len,time_major=False,scope=scope)
            else:
                tf.get_variable_scope().reuse_variables()
                #解码时，通过decoder embedding和decoder bias计算每个词的概率
                output_fn=lambda x:tf.nn.softmax(tf.matmul(x,self.dec_embedding,transpose_b=True)+self.dec_bias)
                decoder_fn=seq2seq.simple_decoder_fn_inference(output_fn=output_fn,encoder_state=encoder_state,embeddings=self.embedding,
                                                               start_of_sequence_id=0,end_of_sequence_id=0,maximum_length=self.subject_len,
                                                               num_decoder_symbols=self.vocab_size,dtype=tf.int32)
                outputs,final_state,final_context_state=seq2seq.dynamic_rnn_decoder(self.decoder_cell,decoder_fn=decoder_fn,inputs=None,sequence_length=self.seq_len,time_major=False,scope=scope)
            
        return outputs,final_state,final_context_state
    def train(self,all_questions,all_subjects,iter_num=10,batch_size=128):
        for i in range(iter_num):
            batch_num=(len(all_questions)+batch_size-1)//batch_size
            total_loss=0
            for batch in range(batch_num):
                questions=all_questions[batch*batch_size:(batch+1)*batch_size]
                subjects=all_subjects[batch*batch_size:(batch+1)*batch_size]
                seq_len=np.zeros(shape=[len(questions),])
                seq_len[:,]=self.subject_len
                feed_dict={self.question_in:questions,self.subject_in:subjects,self.seq_len:seq_len}
                _,loss=self.sess.run([self.opt,self.loss],feed_dict=feed_dict)
                total_loss+=loss
            print("loss:",total_loss/batch_num)
    def predict(self,questions):
        seq_len=np.zeros(shape=[len(questions),])
        seq_len[:,]=self.subject_len
        pre,logits,decoder_outputs=self.sess.run([self.output,self.logits,self.decoder_outptus],feed_dict={self.question_in:questions,self.seq_len:seq_len})
        indices=pre[0][0].indices
        values=pre[0][0].values
        return indices,values,logits,decoder_outputs
    def save_weights(self):
        saver=tf.train.Saver()
        saver.save(self.sess,'./tmp/tmp')
    def load_weights(self):
        saver=tf.train.Saver()
        saver.restore(self.sess,'./tmp/tmp')
            
if __name__=="__main__":
    questions=np.random.randint(low=0,high=1000,size=[20000,20])
    subjects=questions[:,:5]
    
    model=Seq2seqModel()
    model.load_weights()
    model.train(questions,subjects,iter_num=100)
    model.save_weights()
    indices,values,logits,outputs=model.predict(questions)
    pre_ids=np.argmax(logits,axis=-1)
    pprint(pre_ids[:10])
    pprint(subjects[:10])