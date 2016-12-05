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

cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

manifold, indices = cPickle.load(open('manifold.dump', 'rb'))
num_classes = 10
x_data, y_data = cifar10.test.images[indices], cifar10.test.labels[indices]

def colourful_scatter(x_2d, one_hot, num_classes):
# Make a 2D scatter plot with one colour per class
    colors = cm.rainbow(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        subset = np.where(one_hot.argmax(1) == i)[0]
        plt.scatter(x_2d[subset, 0], x_2d[subset, 1], color=colors[i])
    plt.show()

# Plot the manifold
colourful_scatter(manifold, y_data, num_classes)