# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:14:13 2016

@author: liam
"""

import numpy as np
import cPickle
import cifar10_mod
from shutil import rmtree
from os.path import exists
from os import mkdir
import scipy.misc
import time
import matplotlib.cm as cm
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

cifar10, mean_image = cifar10_mod.get_cifar10('cifar10/cifar-10-batches-py')

# Manifold should be an N images * 2 array
# Indices are the indices of the test set that the manifold points represent
# (probably just np.arange(10000))
manifold, indices = cPickle.load(open('manifold.dump', 'rb'))
num_classes = 10
x_data, y_data = cifar10.test.images[indices], cifar10.test.labels[indices]

# Class labels/names
class_labels = y_data.argmax(1)
class_names = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Output all the cifar images we are plotting to .pngs
savedir = 'tmp_imgs/' # WILL BE DELTED FIRST!!!
if exists(savedir):
    rmtree(savedir)
    mkdir(savedir)
else:
    mkdir(savedir)
now = time.time()
img_paths = []
for i, img in enumerate(x_data):
    img_paths.append(savedir + str(i) + '.png')
    scipy.misc.imsave(img_paths[-1], (img + mean_image).astype(np.int))
print(' '.join('Finished exporting CIFAR10 pngs (took', 
      '{:.1f}'.format(time.time() - now), 'seconds)'))

# Colors for the dots based on class
colors = cm.rainbow(np.linspace(0, 1, num_classes))
def rgba_to_hex(rgba):
    return '#%02x%02x%02x' % tuple([int(i*255) for i in rgba[:3]])
colz = [rgba_to_hex(colors[j]) for j in class_labels]

# Aaand plot it
output_file("manifold.html")
source = ColumnDataSource(
        data=dict(
            x=manifold[:, 0],
            y=manifold[:, 1],
            desc=['test set image ' + str(indices[i]) + ': ' + class_names[j]\
                  for i, j in enumerate(class_labels)],
            imgs=img_paths
        )
    )
hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="160" alt="@imgs" width="160"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@desc</span>
                <span style="font-size: 15px; color: #966;">[$index]</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
        """
    )
p = figure(plot_width=1400, plot_height=800, tools=[hover],
           title="Mouse over the dots")
           
p.circle('x', 'y', size=10, source=source, color=colz)
show(p)