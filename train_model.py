from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import ConvNet
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
    
TEST_SIZE = 10000

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
#LOG_DIR_DEFAULT = './logs/cifar10'
LOG_DIR_DEFAULT = './logs/test'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.
    
    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.
    
    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    
    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations
    
    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    tf.reset_default_graph()
    
    sess = tf.Session()
    
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    
    image_shape = cifar10.train.images.shape[1:4]
    num_classes = cifar10.test.labels.shape[1]
    
    x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
    
    model = ConvNet(is_training=is_training, dropout_rate=0.)
    
    logits = model.inference(x)
    
    loss = model.loss(logits, y) + \
           sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.scalar_summary('loss_incl_reg', loss)
    accuracy = model.accuracy(logits, y)
    
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
    
    # Merge all the summaries
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
    
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    # Function for getting feed dicts
    def get_fd(c, train=True):
        if train:
            xd, yd = c.train.next_batch(FLAGS.batch_size)
            return {x : xd, y : yd, is_training : True}
        else:
            xd, yd = c.test.images[:TEST_SIZE], c.test.labels[:TEST_SIZE]
            return {x : xd, y : yd, is_training : False}
    
    saver = tf.train.Saver()
    for epoch in range(0, FLAGS.max_steps):
        if epoch % 100 == 0:
            
          # Print accuracy and loss on test set
          summary, acc, loss_val = \
              sess.run([merged, accuracy, loss], get_fd(cifar10, False))
          test_writer.add_summary(summary, epoch)
          
          print ('\nEpoch', epoch, 
                 '\nTest accuracy:', acc, 
                 '\nTest loss    :', loss_val)
          
          # Save model checkpoint
          if epoch > 0:
              save_path = saver.save(sess, 
                         'checkpoints/conv_basic_epoch' + str(epoch) + '.ckpt')
              print("Model saved in file: %s" % save_path)
    
        # Do training update
        summary, _ = sess.run([merged, train_op], 
                              feed_dict=get_fd(cifar10, True))
        train_writer.add_summary(summary, epoch)
    
    # Print the final accuracy
    summary, acc, loss_val = \
        sess.run([merged, accuracy, loss], get_fd(cifar10, False))
    test_writer.add_summary(summary, epoch + 1)
    print ('\nFinal test accuracy:', acc, '\nFinal test loss    :', loss_val)
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.
    
    Args:
        [optional]
    Returns:
        None
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    tf.reset_default_graph()
    
    sess = tf.Session()
    
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    
    image_shape = cifar10.train.images.shape[1:4]
    num_classes = cifar10.test.labels.shape[1]
    
    x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
    y = tf.placeholder(tf.int32, shape=(None, num_classes), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
    
    model = ConvNet(is_training=is_training, dropout_rate=0.)
    
    logits = model.inference(x)
    
    loss = model.loss(logits, y) + \
           sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.scalar_summary('loss_incl_reg', loss)
    
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    # Function for getting feed dicts
    def get_fd(c, train=True):
        if train:
            xd, yd = c.train.next_batch(FLAGS.batch_size)
            return {x : xd, y : yd, is_training : True}
        else:
            xd, yd = c.test.images[:TEST_SIZE], c.test.labels[:TEST_SIZE]
            return {x : xd, y : yd, is_training : False}
    
    saver = tf.train.Saver()
    
    checkpoint_path = '/home/liam/cloud/y2uni/dl/ass3/checkpoints/'
    checkpoint_file = 'conv_basic_epoch600.ckpt'
    saver.restore(sess, checkpoint_path + checkpoint_file)
    
    # Get the features for the second fully-connected layer
    x_data, y_data = \
        cifar10.test.images[:TEST_SIZE], cifar10.test.labels[:TEST_SIZE]
    features = sess.run([model.hidd2], 
                        {x : x_data, y : y_data, is_training : False})[0]
    
    # Get t-SNE manifold of these features
    tsne = TSNE()
    manifold = tsne.fit_transform(features)
    
    # Save to disk for plotting later
    indices = np.arange(TEST_SIZE)
    cPickle.dump((manifold, indices), open('manifold.dump', 'wb'))
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()

    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()