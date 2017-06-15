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
        self.emb=np.random.rand(self.vocab_size,self.emb_dim)
        
        tf.reset_default_graph()
        
        self.question_in=tf.placeholder(dtype=tf.int32,shape=[None,self.question_len])
        self.subject_in=tf.placeholder(dtype=tf.int32,shape=[None,self.subject_len])
        self.seq_len=tf.placeholder(dtype=tf.int32,shape=[None,])
        
        self.init_weight()
        self.sess=tf.Session()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
    def init_weight(self):
        self.embedding=tf.Variable(self.emb,dtype=self.dtype)
        self.dec_embedding=tf.Variable(self.emb,dtype=self.dtype)
        self.dec_bias=tf.Variable(tf.zeros([self.vocab_size]))
        self.encoder_cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,input_size=None,state_is_tuple=True, activation=tf.nn.tanh)
        self.decoder_cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,input_size=None,state_is_tuple=True,activation=tf.nn.tanh)
        
    def build_model(self):
        questions=tf.nn.embedding_lookup(self.embedding,self.question_in)
        encoder_outputs,encoder_state=self.encoder(questions)
        print("encoder_outputs",encoder_outputs.get_shape())
        decoder_inputs=tf.zeros(shape=[tf.shape(encoder_outputs)[0],self.subject_len,self.hidden_dim])
        decoder_outputs,decoder_state,decoder_context=self.decoder(encoder_state=encoder_state,attention_states=encoder_outputs,inputs=decoder_inputs)
        self.decoder_outptus=decoder_inputs
        print("decoder_outputs:",decoder_outputs.get_shape())
        self.logits,_,_=self.decoder(encoder_state,attention_states=encoder_outputs,inputs=encoder_outputs,is_train=False)
        #beam search解码
        self.output=tf.nn.ctc_greedy_decoder(tf.transpose(self.logits,perm=[1,0,2]),sequence_length=self.seq_len,merge_repeated=False)
        weights=tf.cast(tf.greater(self.subject_in,0),tf.float32)
        
        def softmax_loss_function(inputs,labels):
            '''定义softmax_loss_function损失函数，由seq2seq.sequence_loss调用
            参数：
                1. weights: [num_class,dim],decoder的embedding, 用于进行负采样并计算概率
                2. bias: [num_class],decoder的bias, 用于进行负采样并计算概率
                3. labels: [batch_size,num_true], 正样本的index
                4. inputs: [batch_size,dim],t时刻的输出向量
                5. num_sampled: 采样样本数量
                6. num_classes: 类别总数，即输出词库的大小。
            采样softmax损失函数的计算过程：
                1. 随机生成负样本的index
                2. 使用embedding_lookup操作得到对应的decoder embedding的sampled_weights和sampled_bias：shape分别为[sampled_num,dim],[sampled_num]
                3.使用矩阵乘法，计算每个采样样本的概率：matmul(inputs,sampled_weights)+sampled_bias, 得到shape为[batch_size,num_classes]
                4. 同样，计算正样本的概率
                5. 拼接正样本的概率和负样本的概率，并构造相应的labels
                6. 计算交叉熵损失
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
        '''编码器：调用tf.nn.dynamic_rnn
        '''
        with tf.variable_scope("encoder") as scope:
            initial_state=None#self.encoder_cell.zero_state()
            try:
                outputs,state=tf.nn.dynamic_rnn(self.encoder_cell,inputs,initial_state=initial_state,
                                                dtype=self.dtype,time_major=False,scope=scope)
            except:
                tf.get_variable_scope().reuse_variables()
                outputs,state=tf.nn.dynamic_rnn(self.encoder_cell,inputs,initial_state=initial_state,
                                                dtype=self.dtype,time_major=False,scope=scope)
        return outputs,state
    def decoder(self,encoder_state,attention_states,inputs=None,is_train=True):
        '''
        基于attention的解码器
        1.调用seq2seq.prepare_attention 生成attention的keys/values/functions
        2.训练时，定义dynamic_rnn_decoder用到的attention_decoder_fn_train
        3.预测时，定义dynamic_rnn_decoder用到的attention_decoder_fn_inference
        4.使用以上步骤得到的参数，调用seq2seq.dynamic_rnn_decoder函数
        '''
        with tf.variable_scope("decoder") as scope:
            #1. prepare attention
            keys,values,score_fn,construct_fn=seq2seq.prepare_attention(attention_states=attention_states,attention_option="luong",num_units=self.emb_dim)
            if is_train is True:
                decoder_fn=seq2seq.attention_decoder_fn_train(encoder_state,attention_keys=keys,attention_values=values,attention_score_fn=score_fn,attention_construct_fn=construct_fn,)
                outputs,final_state,final_context_state=seq2seq.dynamic_rnn_decoder(self.decoder_cell,decoder_fn=decoder_fn,inputs=inputs,sequence_length=self.seq_len,time_major=False,scope=scope)
            else:
                tf.get_variable_scope().reuse_variables()
                #解码时，通过decoder embedding和decoder bias计算每个词的概率
                output_fn=lambda x:tf.nn.softmax(tf.matmul(x,self.dec_embedding,transpose_b=True)+self.dec_bias)
                decoder_fn=seq2seq.attention_decoder_fn_inference(output_fn=output_fn,encoder_state=encoder_state,attention_keys=keys,attention_values=values,
                                                                  attention_score_fn=score_fn,attention_construct_fn=construct_fn,embeddings=self.dec_embedding,
                                                                  start_of_sequence_id=0,end_of_sequence_id=1,maximum_length=5,num_decoder_symbols=self.vocab_size)
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
    questions=np.random.randint(low=0,high=1000,size=[2000,20])
    subjects=questions[:,:5]
    
    model=Seq2seqModel()
    #model.load_weights()
    model.train(questions,subjects,iter_num=100)
    model.save_weights()
    indices,values,logits,outputs=model.predict(questions)
    pre_ids=np.argmax(logits,axis=-1)
    pprint(pre_ids[:10])
    pprint(subjects[:10])