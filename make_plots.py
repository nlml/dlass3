# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:14:13 2016

@author: liam
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import cPickle
import cifar10_mod
import os
import itertools

from interactive_manifold import interactive_manifold

cifar10, mean_image = cifar10_mod.get_cifar10('cifar10/cifar-10-batches-py')

#%% BASIC MANIFOLD PLOT
def plot_manifold(manifold, indices, cifar10, title_ext=None, legend=True, save=False):
    num_classes = 10
    y_data = cifar10.test.labels[indices]
    colourful_scatter(manifold, y_data, num_classes, title_ext, legend, save)

def colourful_scatter(x_2d, one_hot, num_classes, title_ext=None, legend=True, save=False):
# Make a 2D scatter plot with one colour per class
    colors = cm.rainbow(np.linspace(0, 1, num_classes))    
    class_labels = one_hot.argmax(1)
    class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(num_classes):
        subset = np.where(class_labels == i)[0]
        plt.scatter(x_2d[subset, 0], x_2d[subset, 1], color=colors[i],
                    label=class_names[i])
    if title_ext is None:
        plt.title('t-SNE manifold')
    else:
        plt.title('t-SNE manifold - ' + title_ext)
    if legend:
        lgd = plt.legend(bbox_to_anchor=(0.99,1), loc=2)
    if save:
        if legend:
            plt.savefig('report/figures/mani_' + title_ext + '.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig('report/figures/mani_' + title_ext + '.png',
                        bbox_inches='tight')
    plt.show()

# Plot the manifold
manifold, indices = cPickle.load(open('manifoldsiamese.dump', 'rb'))
plot_manifold(manifold, indices, cifar10, 'siamese', save=True)
#%% CONVNET PERFORMANCE DROPOUT AND REGULARISATION
dropouts = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
regs = ['%.e' % i for i in [0., 0.1, 0.01, 0.001, 0.0001]]
cols = [i for i in itertools.product(dropouts, regs)]
epoch_lens = [i*100 for i in range(151)]

directory = 'sara_outputs/'

for loss_or_acc in ['accuracy', 'loss']:
    
    scores_df = pd.DataFrame(index=epoch_lens,
                             columns=pd.MultiIndex.from_tuples(cols,
                                         names=['dropout', 'reg']))
    scores_df.index.name = 'epoch'
    
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
                dropout, reg = parse_filename(filename)
                if dropout < 0.9:
                    for r in f:
                        while '  ' in r:
                            r = r.replace('  ', ' ')
                        r = r.replace(' :', ':')
                        if 'Epoch' in r:
                            curr_epoch = int(r.replace('Epoch', '').strip())
                        elif 'Test ' + loss_or_acc in r:
                            acc = float(r.split(' ')[2])
                            scores_df.loc[curr_epoch][dropout]['%.e' % reg] = acc
                        elif 'Final test ' + loss_or_acc in r:
                            acc = float(r.split(' ')[3])
                            scores_df.loc[curr_epoch + 100][dropout]['%.e' % reg] = acc
    
    scores_df.columns.names = ['dropout', 'reg']
    scores_df = scores_df.stack()
    reg_idx = scores_df.index.get_level_values('reg')
    for i, rg in enumerate(reg_idx.unique()):
        plt.figure(figsize=(8,6))
        tmp_df = scores_df[reg_idx==rg]
        tmp_df.index = tmp_df.index.get_level_values('epoch')
        tmp_df.plot()
        plt.xlabel('Epoch')
        plt.title(loss_or_acc + ' on test set, l2 reg strength = ' + str(rg))
        plt.savefig('report/figures/' + loss_or_acc + '_reg' + str(rg) + '.png')
        plt.show()
        
    plt.figure(figsize=(14,10))
    sns.barplot(x='dropout', y=0, hue='reg', 
                data=scores_df.unstack().max().reset_index())
    plt.legend(bbox_to_anchor=(0.13, 0.4), loc=0, title='weight reg', fontsize=14)
    plt.ylabel('best ' + loss_or_acc + ' over all epochs')
    plt.ylim((0.7,.8))
    plt.savefig('report/figures/' + loss_or_acc + '_bar.png')
    plt.show()
#%% SIAMESE TRAINING PLOT
scores_df = pd.DataFrame(index=[i*100 for i in range(1000)], columns=['accuracy', 'loss'])
for loss_or_acc in ['accuracy', 'loss']:
    
    with open(os.path.join('checkpoints_siamese/margin0.1_steps60k_frac0.2_reg0.0_dropout0.0_.output'),
              'r') as f:
        f = f.readlines()
        f = [i.replace('\r\n', '') for i in f]
        curr_epoch = -1
        for r in f:
            while '  ' in r:
                r = r.replace('  ', ' ')
            r = r.replace(' :', ':')
            if 'Epoch' in r:
                curr_epoch = int(r.replace('Epoch', '').strip())
            elif 'Validation ' + loss_or_acc in r:
                acc = float(r.split(' ')[2])
                scores_df.loc[curr_epoch][loss_or_acc] = acc
            elif 'Final validation ' + loss_or_acc in r:
                acc = float(r.split(' ')[3])
                scores_df.loc[curr_epoch + 100][loss_or_acc] = acc

for loss_or_acc in ['accuracy', 'loss']:         
    scores_df[loss_or_acc][:501].plot()
    plt.title('Siamese ' + loss_or_acc + ' on validation set (margin = 0.1)')
    if loss_or_acc == 'loss':
        plt.savefig('report/figures/siamese_loss.png')
    plt.show()
#%% REFINE AFTER K PLOT
ks = [-1, 0, 100, 1000, 2500]

scores_df = pd.DataFrame(index=[i*100 for i in range(101)], columns=ks)
scores_df.index.name = 'epoch'
directory = 'vgg_outputs/'

def parse_filename(filename):
    k = filename.replace('vgg_k', '').replace('.output', '')
    return int(k)
    
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as f:
        f = f.readlines()
        f = [i.replace('\r\n', '') for i in f]
        curr_epoch = -1
        k = parse_filename(filename)
        for r in f:
            if 'Epoch' in r:
                curr_epoch = int(r.replace('Epoch', '').strip())
            elif 'Test accur' in r:
                acc = float(r.split(' ')[2])
                scores_df.loc[curr_epoch][k] = acc
            elif 'Final test accur' in r:
                acc = float(r.split(' ')[3])
                scores_df.loc[curr_epoch + 100][k] = acc

scores_df.columns.names = ['k']
scores_df.columns = ['VGG weights fixed' if i == -1 else 'Refine after ' + str(i) for i in scores_df.columns]
scores_df.plot(figsize=(14,10))
plt.savefig('report/figures/refine_after_k.png')
plt.ylabel('accuracy on test set')
plt.show()
#%% LINEAR MODELS ON EXTRACTED FEATURES USING VARIOUS REGULARISATION SETTINGS
manifolds, results_df = cPickle.load(open('manifoldses.dump', 'rb'))
tmp_df = results_df.stack().unstack('epoch').max(1).reset_index()
tmp_df.columns = ['dropout', 'reg', 'type', 'accuracy']
sns.set(font_scale = 1.6)
pals = ['Reds', 'Blues', 'Greens', 'Greys']
num_reg_cats = len(tmp_df.reg.unique())
plt.figure(figsize=(16,10))
for i, layer_name in enumerate(['fc2', 'fc1', 'flat', 'flat']):
    tdf = tmp_df[tmp_df.type==layer_name]
    if i == 3:
        tdf.accuracy = 0.
    ax = sns.barplot(x='dropout', y='accuracy', hue='reg', 
                     data=tdf, palette=pals[i], alpha=0.8)
                     
plt.title('Maximum accuracy achieved on test set, features extracted from flatten vs. fc1 vs. fc2 layers')
plt.ylabel('max accuracy achieved ')

handles, labels = ax.get_legend_handles_labels()
types = ['fc2', 'fc1', 'flat']
i = 0
hands, labs = [], []
hands += handles[3*num_reg_cats:]
labs += labels[3*num_reg_cats:]
for j in range(num_reg_cats - 2, 3*num_reg_cats, num_reg_cats):
    hands.append(handles[j])
    labs.append(labels[j])
    labs[-1] = types[i]
    i += 1
h = handles[-num_reg_cats-1]
for p in h.patches:
    p.set_visible(False)
l = ''
for i in range(2):
    hands.append(h)
    labs.append(l)
plt.legend(hands, labs, bbox_to_anchor=(0.24, 0.72), loc=0, ncol=2, 
           title='weight reg       layer     ')
plt.ylim((0.4,0.8))
plt.savefig('report/figures/lm_acc.png')
plt.show()
#%%
def mani_best(best):
    best = best.reset_index().iloc[:, :3].values.reshape(-1)
    best[2] = float(best[2])
    m, i = manifolds[best[1]][best[2]][best[0]]
    return m, i, best
    
def plot_a_mani(m, i, best, app, size=1000):
    print (m.shape)
    plt.figure(figsize=(10,8))
    full_title = app + ' config (' + 'dropout = ' + str(best[1]) + ' / reg = ' + '%.e' % best[2] + ')'
    plot_manifold(m, i, cifar10, full_title)
    plt.show()
    subset = np.random.choice(np.arange(m.shape[0]), size, replace=False)
    interactive_manifold(m[subset, :], i[subset], cifar10, mean_image, app, full_title)
    
def plot_mani_best(best, app, size=1000):
    m, i, best = mani_best(best)
    plot_a_mani(m, i, best, app, size)

manifolds_basic = cPickle.load(open('manifolds_basic.dump', 'rb'))
for name in types[-1::-1]:
    m, i = manifolds_basic[name]
    plot_manifold(m, i, cifar10, name, legend=name=='fc2', save=True)
#%%
plot_mani_best(results_df[results_df.fc2==results_df.fc2.max()], 'Best')
plot_mani_best(results_df[results_df.fc2==results_df.fc2.min()], 'Worst')