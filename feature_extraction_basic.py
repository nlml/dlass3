from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from itertools import product
import pandas as pd
from os.path import exists
import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import ConvNet
from sklearn.manifold import TSNE
import cPickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from time import sleep
    
tf.reset_default_graph()


def parse_filename(filename):
    filename = filename.replace('checkpoints_', '')
    if 'dropout' not in filename:
        dropout = 0.0
    else:
        dropout = filename[filename.find('dropout') + 7:]
        dropout = float(dropout[:dropout.find('.')])/10
    if filename.startswith('1'):
        reg = 0.1
    else:
        reg = float('0.' + filename[:filename.find('reg')])
    return dropout, reg

cifar10     = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')

print ('done getting cifar10')

image_shape = cifar10.train.images.shape[1:4]
num_classes = cifar10.test.labels.shape[1]

# Construct linear convnet graph
x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
y = tf.placeholder(tf.int32, shape=(None, num_classes), name='y')
is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')

model = ConvNet(is_training=is_training)

_ = model.inference(x)
#%%
with tf.Session() as sess:
    
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    checkpoint_dirs = ['checkpoints_0reg_lr1e4_sqrtinit']
    ckpt_file = 'epoch14000.ckpt'
    #subdir = './'
    subdir = 'checkpoints_new/'
    test_size = 1000
    
    saver = tf.train.Saver()
    
    for ckpt_path in checkpoint_dirs:
        
        ckpt_path_use = subdir + ckpt_path + '/'
        if not exists(ckpt_path_use + ckpt_file):
            print('skipping', ckpt_path_use + ckpt_file)
            continue
        
        # Restore checkpoint
        saver.restore(sess, ckpt_path_use + ckpt_file)
        
        # Get testing data for feed dict
        x_data_test, y_data_test = \
            cifar10.test.images[:test_size], cifar10.test.labels[:test_size]
            
        # Get the test set features at flatten, fc1 and fc2 layers
        flatten_features_test, fc1_features_test, fc2_features_test = \
            sess.run([model.flatten, model.fc1, model.fc2],
                     {x : x_data_test, y : y_data_test, is_training : False})
                     
        # Save to disk for plotting later
        indices = np.arange(test_size)
        
        features_list = [['flat', flatten_features_test],
                         ['fc1', fc1_features_test],
                         ['fc2', fc2_features_test]]
                         
        manifolds_out = {}
        for (name, features_test) in features_list:
            print('tsne..')
            # Get t-SNE manifold of these features
            tsne = TSNE()
            manifolds_out[name] = (tsne.fit_transform(features_test), indices)
            print('done')
        
        cPickle.dump(manifolds_out, open('manifolds_basic.dump', 'wb'))