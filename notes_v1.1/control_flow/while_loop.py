#coding:utf-8
import tensorflow as tf

output=tf.zeros(shape=[3,4])
outputs=[output]

def step(inputs):
    return inputs+1
    

for i in range(5):
    output=step(outputs[-1])
    outputs.append(output)
    
with tf.Session() as sess:
    out=sess.run(outputs)
    print(out)