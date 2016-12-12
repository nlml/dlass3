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

checkpoint_dirs = [
'checkpoints_0001reg_lr1e4_sqrtinit',
'checkpoints_0001reg_lr1e4_sqrtinit_dropout10',
'checkpoints_0001reg_lr1e4_sqrtinit_dropout20',
'checkpoints_0001reg_lr1e4_sqrtinit_dropout30',
'checkpoints_0001reg_lr1e4_sqrtinit_dropout40',
'checkpoints_0001reg_lr1e4_sqrtinit_dropout50',
'checkpoints_001reg_lr1e4_sqrtinit',
'checkpoints_001reg_lr1e4_sqrtinit_dropout10',
'checkpoints_001reg_lr1e4_sqrtinit_dropout20',
'checkpoints_001reg_lr1e4_sqrtinit_dropout30',
'checkpoints_001reg_lr1e4_sqrtinit_dropout40',
'checkpoints_001reg_lr1e4_sqrtinit_dropout50',
'checkpoints_01reg_lr1e4_sqrtinit',
'checkpoints_01reg_lr1e4_sqrtinit_dropout10',
'checkpoints_01reg_lr1e4_sqrtinit_dropout20',
'checkpoints_01reg_lr1e4_sqrtinit_dropout30',
'checkpoints_01reg_lr1e4_sqrtinit_dropout40',
'checkpoints_01reg_lr1e4_sqrtinit_dropout50',
'checkpoints_0reg_lr1e4_sqrtinit',
'checkpoints_0reg_lr1e4_sqrtinit_dropout10',
'checkpoints_0reg_lr1e4_sqrtinit_dropout20',
'checkpoints_0reg_lr1e4_sqrtinit_dropout30',
'checkpoints_0reg_lr1e4_sqrtinit_dropout40',
'checkpoints_0reg_lr1e4_sqrtinit_dropout50',
'checkpoints_1reg_lr1e4_sqrtinit',
'checkpoints_1reg_lr1e4_sqrtinit_dropout10',
'checkpoints_1reg_lr1e4_sqrtinit_dropout20',
'checkpoints_1reg_lr1e4_sqrtinit_dropout30',
'checkpoints_1reg_lr1e4_sqrtinit_dropout40',
'checkpoints_1reg_lr1e4_sqrtinit_dropout50'
]

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

fc2 = model.fc2
#%%
train_size_lm = 2000
test_size = 1000
ckpt_file_in = 'epoch14000.ckpt'
#subdir = './'
subdir = 'checkpoints_new/'

dropouts = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
regs = ['%.e' % i for i in [0., 0.1, 0.01, 0.001, 0.0001]]
epochs = np.arange(10000, 15000, 1000)
cols = [i for i in product(epochs, dropouts, regs)]
results_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(cols, names=['epoch', 'dropout', 'reg']),
                          columns=['flat', 'fc1', 'fc2'])

manifolds = {}
#%%
with tf.Session() as sess:
    
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    # Restore checkpoint
    saver = tf.train.Saver()
    
    while results_df.isnull().sum().sum()>0:
        #sleep(3)
        for ckpt_path in checkpoint_dirs:
            
            for epoch in epochs:
                
                ckpt_file = ckpt_file_in.replace('14000', str(epoch))
            
                dropout, reg = parse_filename(ckpt_path)
                
                if dropout not in manifolds:
                    manifolds[dropout] = {}
                
                if reg not in manifolds[dropout]:
                    manifolds[dropout][reg] = {}
                
                ckpt_path_use = subdir + ckpt_path + '/'
                if not exists(ckpt_path_use + ckpt_file):
                    print('skipping', ckpt_path_use + ckpt_file)
                    continue

                if results_df.loc[(epoch, dropout, '%.e' % reg)].isnull().sum() == 0:
                    #print('already done', ckpt_path_use + ckpt_file)
                    continue
                
                saver.restore(sess, ckpt_path_use + ckpt_file)
                
                # Get testing data for feed dict
                x_data_test, y_data_test = \
                    cifar10.test.images[:test_size], cifar10.test.labels[:test_size]
                    
                # Get the test set features at flatten, fc1 and fc2 layers
                flatten_features_test, fc1_features_test, fc2_features_test = \
                    sess.run([model.flatten, model.fc1, fc2], 
                             {x : x_data_test, y : y_data_test, is_training : False})
                print('tsne..')
                # Get t-SNE manifold of these features
                tsne = TSNE()
                manifold = tsne.fit_transform(fc2_features_test)
                print('done')
                # Save to disk for plotting later
                indices = np.arange(test_size)
                
                manifolds[dropout][reg][epoch] = (manifold, indices)
                
                # Get training data for feed dict
                x_data_train, y_data_train = \
                    cifar10.train.images[:train_size_lm], cifar10.train.labels[:train_size_lm]
                    
                # Get train set features at flatten, fc1 and fc2 layers
                flatten_features_train, fc1_features_train, fc2_features_train = \
                    sess.run([model.flatten, model.fc1, fc2], 
                             {x : x_data_train, y : y_data_train, is_training : False})
                
                features_list = [['flat', flatten_features_train, flatten_features_test],
                                 ['fc1', fc1_features_train, fc1_features_test],
                                 ['fc2', fc2_features_train, fc2_features_test]]
                print ('dropout:', dropout, 'reg:', reg)
                now = time.time()
                for (name, features_train, features_test) in features_list:
                    classif = OneVsRestClassifier(SVC(kernel='linear'))
                    classif.fit(features_train, y_data_train)
                    lm_test_predictions = classif.predict(features_test)
                    acc = np.mean(np.equal(np.argmax(y_data_test, 1), np.argmax(lm_test_predictions, 1)))
                    print (name, 'accuracy =', np.round(acc*100, 2), '%')
                    results_df.loc[(epoch, dropout, '%.e' % reg)][name] = acc
                    
                print(results_df)
                cPickle.dump((manifolds, results_df), open('manifoldses.dump', 'wb'))