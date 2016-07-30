# -*- coding: utf-8 -*-
#minist_data
import tensorflow as tf
import os
import gzip
from six.moves import urllib
import cPickle as pickle
import numpy as np

def load_data(dataset='mnist.pkl.gz'):
    new_path=os.path.join(os.path.split(__file__)[0],dataset)
    if not os.path.isfile(new_path):
        origin='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' %origin)
        urllib.requestion.urlretrieve(origin,new_path)
    #加载数据
    with gzip.open(new_path) as f:
        train_set,valid_set,test_set=pickle.load(f)
    return train_set,valid_set,test_set
    
def dense_to_one_hot(labels_desnse,num_calsses=10):
    '''把scalar的类标签转换成one-hot向量'''
    num_labels=labels_desnse.shape[0]
    labels_one_hot=np.zeros((num_labels,num_calsses),dtype="float32")
    labels_one_hot[np.arange(num_labels),labels_desnse]=1.0
    return labels_one_hot
class DataSet(object):
    def __init__(self,images,labels,fake_data=False):
        if fake_data:
            self._num_examples=10000
        else:
            assert images.shape[0]==labels.shape[0],(
                "image.shape:%s labels.shape: %s" % (images.shape,labels.shape)
                 )
            self._num_examples=images.shape[0]
            #shape从[num examples, rows, columns, depth]
            #转为[num examples, rows*columns] (假设depth=1)
            assert images.shape[3]==1
            images=images.reshape(images.shape[0],images.shape[1]*images.shape[2])
            #从[0,255]转换为[0,1.0]
            images=images.astype(np.float32)
            images=np.multiply(images,1.0/255.0)
        self._images=images
        self._labels=labels
        self._epochs_completed=0
        self._index_in_epoch=0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self,batch_size,fake_data=False):
        if fake_data:
            fake_image=[1.0 for _ in range(784)]
            fake_label=0
            return [fake_image for _ in xrange(batch_size)],[fake_label for _  in xrange(batch_size)]
        start=self._index_in_epoch
        self._index_in_epoch+=batch_size
        if self._index_in_epoch >self._num_examples:
            #结束epoch
            self._epochs_completed+=1
            #Shuffle the data
            perm=np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images=self._images[perm]
            self._labels=self._labels[perm]
            #开始下一个epoch
            start=0
            self._index_in_epoch=batch_size
            assert batch_size<=self._num_examples
        end=self._index_in_epoch
        return self._images[start:end],self._labels[start:end]

def read_data_sets(fake_data=False,one_hot=False):
    class DataSets(object):
        pass
    data_sets=DataSets()
    if fake_data:
        data_sets.train=DataSet([],[],fake_data=True)
        data_sets.validation=DataSet([],[],fake_data=True)
        data_sets.test=DataSet([],[],fake_data=True)
        return data_sets
    train_data,validation_data,test_data=load_data()
    train_images,train_labels=train_data
    train_images=np.asarray(np.reshape(train_images,(train_images.shape[0],28,28,1)),dtype='float32')
    if one_hot:
        train_labels=dense_to_one_hot(train_labels)

    validation_images,validation_labels=validation_data
    validation_images=np.asarray(np.reshape(validation_images,(validation_images.shape[0],28,28,1)),dtype="float32")
    if one_hot:
         validation_labels=dense_to_one_hot(validation_labels)

    test_images,test_labels=test_data
    test_images=np.asarray(np.reshape(test_images,(test_images.shape[0],28,28,1)),dtype="float32")
    if one_hot:
        test_labels=dense_to_one_hot(test_labels)

    data_sets.train=DataSet(train_images,train_labels)
    data_sets.validation=DataSet(validation_images,validation_labels)
    data_sets.test=DataSet(test_images,test_labels)
    return data_sets





if __name__=="__main__":
    train_set,valid_set,test_set=load_data()
    x=train_set[0]
    y=train_set[1]
    one_hot=dense_to_one_hot(y)
    print(x.shape)
    
    data_sets=read_data_sets(one_hot=True)
    train=data_sets.train
    print train.next_batch(20)

