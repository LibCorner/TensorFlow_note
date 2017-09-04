#coding:utf-8
import tensorflow as tf
import gym
import numpy as np

#创建CartPole问题的环境
env=gym.make("CartPole-v0")
#初始化环境
env.reset()

random_episodes=0
reward_sum=0
while random_episodes<10:
    env.render()
    observation,reward,done,_=env.step(np.random.randint(0,2))
    #累计奖励
    reward_sum+=reward
    if done:
        random_episodes+=1
        print("Reward for this episode was:",reward_sum)
        reward_sum=0
        env.reset()
        
#MLP层，隐藏节点数为10， 环境信息维度为4
H=10
batch_size=5
learning_rate=1e-2
gamma=0.99 #衰减系数
D=4 #输入维度

observations=tf.placeholder(tf.float32,[None,D],name='input_x')
W1=tf.get_variable('W1',shape=[D,H],
                   initializer=tf.contrib.layers.xavier_initializer())
layer1=tf.nn.relu(tf.matmul(observations,W1))
W2=tf.get_variable("W2",shape=[H,1],
                   initializer=tf.contrib.layers.xavier_initializer())
score=tf.matmul(layer1,W2)
probability=tf.nn.sigmoid(score)
