# -*- coding: utf-8 -*-
#minist_data
import tensorflow as tf
import os
import gzip
from six.moves import urllib
import cPickle as pickle

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

if __name__=="__main__":
    train_set,valid_set,test_set=load_data()


