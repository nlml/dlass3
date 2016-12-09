# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:14:13 2016

@author: liam
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import cPickle
import cifar10_utils
import os

cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

manifold, indices = cPickle.load(open('manifold.dump', 'rb'))
num_classes = 10
x_data, y_data = cifar10.test.images[indices], cifar10.test.labels[indices]

def colourful_scatter(x_2d, one_hot, num_classes):
# Make a 2D scatter plot with one colour per class
    colors = cm.rainbow(np.linspace(0, 1, num_classes))    
    class_labels = one_hot.argmax(1)
    for i in range(num_classes):
        subset = np.where(class_labels == i)[0]
        plt.scatter(x_2d[subset, 0], x_2d[subset, 1], color=colors[i])
    plt.title('t-SNE manifold')
    plt.show()

# Plot the manifold
colourful_scatter(manifold, y_data, num_classes)

#%%
import itertools
dropouts = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
regs = ['%.e' % i for i in [0., 0.1, 0.01, 0.001, 0.0001]]
cols = [i for i in itertools.product(dropouts, regs)]

scores_df = pd.DataFrame(index=[i*100 for i in range(150)],
                         columns=pd.MultiIndex.from_tuples(cols,
                                     names=['dropout', 'reg']))
scores_df.index.name = 'epoch'
directory = 'sara_outputs/'

def parse_filename(filename):
    if 'dropout' not in filename:
        dropout = 0.0
    else:
        dropout = filename[filename.find('dropout') + 7:]
        dropout = float(dropout[:dropout.find('.')])/100
    if filename.startswith('1'):
        reg = 0.1
    else:
        reg = float('0.' + filename[:filename.find('reg')])
    return dropout, reg

for filename in os.listdir(directory):
    if 'lr1e4' in filename:
        with open(os.path.join(directory, filename), 'r') as f:
            f = f.readlines()
            f = [i.replace('\r\n', '') for i in f]
            curr_epoch = -1
            accs = []
            dropout, reg = parse_filename(filename)
            if dropout < 0.9:
                for r in f:
                    if 'Epoch' in r:
                        curr_epoch = int(r.replace('Epoch', '').strip())
                    elif 'Test accur' in r:
                        acc = float(r.split(' ')[2])
                        scores_df.loc[curr_epoch][dropout, '%.e' % reg] = acc

scores_df = pd.DataFrame.from_dict(scores_df)

scores_df.columns.names = ['dropout', 'reg']
scores_df = scores_df.stack()
drop_idx = scores_df.index.get_level_values('reg')
for i, dr in enumerate(drop_idx.unique()):
    scores_df[drop_idx==dr].plot()
    plt.title('Accuracy on test set, l2 reg strength = ' + str(dr))
    plt.show()
plt.show()

plt.figure()
seaborn.barplot(x='dropout', y=0, hue='reg', 
                data=scores_df.unstack().max().reset_index())
plt.show()