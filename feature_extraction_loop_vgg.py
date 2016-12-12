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

cifar10     = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')

print ('done getting cifar10')

image_shape = cifar10.train.images.shape[1:4]
num_classes = cifar10.test.labels.shape[1]

tf.reset_default_graph()
# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')

y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
stop_the_gradient = tf.placeholder(dtype=tf.bool, shape=(), name='stopGrad')
is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')

from vgg import load_pretrained_VGG16_pool5
from vgg_readout import VGGReadOut

pool5, assign_ops = load_pretrained_VGG16_pool5(x)
pool5 = tf.cond(stop_the_gradient, lambda: tf.stop_gradient(pool5), lambda: pool5)

model = VGGReadOut(is_training=is_training)
                   
# Get logits, loss, accuracy, train optimisation step
logits   = model.inference(pool5, 'sqrt')

# For saving checkpoints
saver = tf.train.Saver()

#%%
train_size_lm = 1000
test_size = 1000
ckpt_file_in = 'epochEPP_kKVAL.ckpt'
#subdir = './'
subdir = 'checkpoints_vgg/'

ks = [-1, 0]
epochs=[10000]
cols = [i for i in product(epochs, ks)]
results_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(cols, 
                                                          names=['epoch', 'k']),
                          columns=['flat', 'fc1', 'fc2'])

manifolds = {}
#%%
ckpt_path = 'checkpoints_vgg'
with tf.Session() as sess:
    
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    # Restore checkpoint
    saver = tf.train.Saver()
    
    for k in ks:
        
        for epoch in epochs:
            
                        
            ckpt_file = ckpt_file_in.replace('EPP', str(epoch)).replace('KVAL', str(k))
        
            if k not in manifolds:
                manifolds[k] = {}
            if epoch not in manifolds[k]:
                manifolds[k][epoch] = {}
                
            ckpt_path_use = ckpt_path + '/'
            if not exists(ckpt_path_use + ckpt_file):
                print('skipping', ckpt_path_use + ckpt_file)
                continue

            if results_df.loc[(epoch, k)].isnull().sum() == 0:
                print('already done', ckpt_path_use + ckpt_file)
                continue
            
            saver.restore(sess, ckpt_path_use + ckpt_file)
            
            # Get testing data for feed dict
            x_data_test, y_data_test = \
                cifar10.test.images[:test_size], cifar10.test.labels[:test_size]
                
            # Get the test set features at flatten, fc1 and fc2 layers
            flatten_features_test, fc1_features_test, fc2_features_test = \
                sess.run([model.flatten, model.fc1, model.fc2], 
                         {x : x_data_test, y : y_data_test, is_training : False, stop_the_gradient : False})
                         
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
                manifold = tsne.fit_transform(features_test)
                manifolds[k][epoch][name] = (manifold, indices)
                print('done')
            
                
            # Get training data for feed dict
            x_data_train, y_data_train = \
                cifar10.train.images[:train_size_lm], cifar10.train.labels[:train_size_lm]
                
            # Get train set features at flatten, fc1 and fc2 layers
            flatten_features_train, fc1_features_train, fc2_features_train = \
                sess.run([model.flatten, model.fc1, model.fc2], 
                         {x : x_data_train[train_size_lm/2:], y : y_data_train[train_size_lm/2:], 
                          is_training : False, stop_the_gradient : False})
            # Get train set features at flatten, fc1 and fc2 layers
            flatten_features_train2, fc1_features_train2, fc2_features_train2 = \
                sess.run([model.flatten, model.fc1, model.fc2], 
                         {x : x_data_train[:train_size_lm/2], y : y_data_train[:train_size_lm/2], 
                          is_training : False, stop_the_gradient : False})
                          
            flatten_features_train, fc1_features_train, fc2_features_train =\
                np.vstack((flatten_features_train, flatten_features_train2)),\
                np.vstack((fc1_features_train, fc1_features_train2)), \
                np.vstack((fc2_features_train, fc2_features_train2))
            
            features_list = [['flat', flatten_features_train, flatten_features_test],
                             ['fc1', fc1_features_train, fc1_features_test],
                             ['fc2', fc2_features_train, fc2_features_test]]
            print ('k:', k, 'epoch:', epoch)
            now = time.time()
            for (name, features_train, features_test) in features_list:
                classif = OneVsRestClassifier(SVC(kernel='linear'))
                classif.fit(features_train, y_data_train)
                lm_test_predictions = classif.predict(features_test)
                acc = np.mean(np.equal(np.argmax(y_data_test, 1), np.argmax(lm_test_predictions, 1)))
                print (name, 'accuracy =', np.round(acc*100, 2), '%')
                results_df.loc[(epoch, k)][name] = acc
                
            print(results_df)
            cPickle.dump((manifolds, results_df), open('manifolds_vgg.dump', 'wb'))