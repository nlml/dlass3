from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

import cifar10_vgg as cifar10_utils

from vgg import load_pretrained_VGG16_pool5
from vgg_readout import VGGReadOut

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 10000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10_vgg'
CHECKPOINT_DIR_DEFAULT = './checkpoints_vgg'

### NEW ###
tf.reset_default_graph()
TEST_SIZE_DEFAULT = 200
###########
    
def standard_cifar10_get(FLAGS):
    cifar10         = cifar10_utils.get_cifar10(FLAGS.data_dir)
    image_shape     = cifar10.train.images.shape[1:4]
    num_classes     = cifar10.test.labels.shape[1]
    return cifar10, image_shape, num_classes 

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
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

#==============================================================================
# def feature_extraction():
#     """
#     Performs training and evaluation of your model.
#     
#     First define your graph using vgg.py with your fully connected layer.
#     Then define necessary operations such as trainer (train_step in this case),
#     savers and summarizers. Finally, initialize your model within a
#     tf.Session and do the training.
#     
#     ---------------------------------
#     How often to evaluate your model:
#     ---------------------------------
#     - on training set every PRINT_FREQ iterations
#     - on test set every EVAL_FREQ iterations
#     
#     ---------------------------
#     How to evaluate your model:
#     ---------------------------
#     Evaluation on test set should be conducted over full batch, i.e. 10k images,
#     while it is alright to do it over minibatch for train set.
#     """
# 
#     # Set the random seeds for reproducibility. DO NOT CHANGE.
#     tf.set_random_seed(42)
#     np.random.seed(42)
#     
#     ########################
#     # PUT YOUR CODE HERE  #
#     ########################
#     
#     # Cifar10 stuff
#     cifar10, image_shape, num_classes = standard_cifar10_get(FLAGS)
#     
#     tf.reset_default_graph()
#     # Placeholder variables
#     x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
#     y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
#     is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
#     
#     pool5, assign_ops, kernel = load_pretrained_VGG16_pool5(x)
#     print (pool5.get_shape())
#     
#     model = VGGReadOut(is_training=is_training, 
#                        dropout_rate=FLAGS.dropout_rate, 
#                        save_stuff=FLAGS.save_stuff, 
#                        fc_reg_str=FLAGS.fc_reg_str)
#                        
#     # Get logits, loss, accuracy, train optimisation step
#     logits   = model.inference(pool5, w_init=FLAGS.w_init)
#     reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#     loss     = model.loss(logits, y) + reg_loss
#     tf.scalar_summary('loss_incl_reg', loss)
#     
#     # Function for getting feed dicts
#     def get_feed(c, train=True):
#         if train:
#             xd, yd = c.train.next_batch(FLAGS.batch_size)
#             return {x : xd, y : yd, is_training : True}
#         else:
#             xd, yd = c.test.images[:FLAGS.test_size], c.test.labels[:FLAGS.test_size]
#             return {x : xd, y : yd, is_training : False}
#     
#     from sklearn.ensemble import RandomForestClassifier
#     #%%
#     LM_MODEL_NTRAIN = 10000
#     LM_MODEL_NTEST = 10000
#     
#     chunks = 10
#     chunksize_train = int(LM_MODEL_NTRAIN * (1. / chunks))
#     chunksize_test = int(LM_MODEL_NTEST * (1. / chunks))
#     x_train = []
#     x_test  = []
#     #%%
#     with tf.Session() as sess:
#         
#         # Initialise all variables
#         tf.initialize_all_variables().run(session=sess)
#         
#         for oppy in assign_ops:
#             sess.run(oppy)
#             
#     #==============================================================================
#     #     img = cifar10.train.images[0:1]
#     #     print(img)
#     #     print(sess.run([pool5], {x : img})[0])
#     #==============================================================================
#     
#         for st in range(0, LM_MODEL_NTRAIN, chunksize_train):
#             en = st + chunksize_train
#             x_train.append(sess.run([pool5], {x : cifar10.train.images[st:en]})[0])
#             
#         for st in range(0, LM_MODEL_NTEST, chunksize_test):
#             en = st + chunksize_test
#             x_test.append(sess.run([pool5], {x : cifar10.test.images[st:en]})[0])
#         
#     #%%
#     x_train = np.vstack(x_train).reshape(LM_MODEL_NTRAIN, -1)
#     x_test = np.vstack(x_test).reshape(LM_MODEL_NTEST, -1)
#     #%%
#     y_train = cifar10.train.labels[:LM_MODEL_NTRAIN].argmax(1)
#     y_test = cifar10.test.labels[:LM_MODEL_NTEST].argmax(1)
#     #%%
#     from sklearn.neural_network import MLPClassifier
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.svm import SVC
#     
#     classif = OneVsRestClassifier(SVC(kernel='linear'))
#     classif.fit(x_train, y_train)
#     lm_test_predictions = classif.predict(x_test)
#     acc = np.mean(np.argmax(y_test, 1)==np.argmax(lm_test_predictions, 1))
#     print (name, 'accuracy =', np.round(acc*100, 2), '%')
#     #%%
#     #rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
#     rf = MLPClassifier(verbose=True, hidden_layer_sizes=(384,192,64), 
#                        tol=0.1, batch_size=128, alpha=0.1).fit(x_train, y_train)
#     preds = rf.predict(x_test)
#     print (np.mean(preds==y_test))
#     preds = rf.predict(x_train)
#     print (np.mean(preds==y_train))
#     assert 1==0
#     #%%
#     from sklearn.manifold import TSNE
#     import cPickle
#     # Get t-SNE manifold of these features
#     tsne = TSNE()
#     manifold = tsne.fit_transform(x_test)
#     
#     # Save to disk for plotting later
#     indices = np.arange(LM_MODEL_NTEST)
#     cPickle.dump((manifold, indices), open('manifold_vgg.dump', 'wb'))
#==============================================================================
#%%
def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    # Cifar10 stuff
    cifar10, image_shape, num_classes = standard_cifar10_get(FLAGS)
    
    tf.reset_default_graph()
    # Placeholder variables
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
    print('imgshape', image_shape)
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
    stop_the_gradient = tf.placeholder(dtype=tf.bool, shape=(), name='stopGrad')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
    
    if FLAGS.refine_after_k == -1:
        refine_k = FLAGS.max_steps + 1
    else:
        refine_k = FLAGS.refine_after_k
    
    pool5, assign_ops = load_pretrained_VGG16_pool5(x)
    pool5 = tf.cond(stop_the_gradient, lambda: tf.stop_gradient(pool5), lambda: pool5)

    model = VGGReadOut(is_training=is_training, 
                       dropout_rate=FLAGS.dropout_rate, 
                       save_stuff=FLAGS.save_stuff, 
                       fc_reg_str=FLAGS.fc_reg_str)
                       
    # Get logits, loss, accuracy, train optimisation step
    logits   = model.inference(pool5, w_init=FLAGS.w_init)
    accuracy = model.accuracy(logits, y)
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss     = model.loss(logits, y) + reg_loss
    tf.scalar_summary('loss_incl_reg', loss)
    train_op = train_step(loss)
    
    # Function for getting feed dicts
    def get_feed(c, train=True, epoch=-1):
        if epoch==-1:
            raise Exception('I need to know what epoch it is to decide ' + \
                            'whether to stop the gradient!')
        if train:
            xd, yd = c.train.next_batch(FLAGS.batch_size)
            return {x : xd, y : yd, is_training : True, 
                    stop_the_gradient: epoch < refine_k}
        else:
            xd, yd = c.test.images[:FLAGS.test_size], \
                     c.test.labels[:FLAGS.test_size]
            return {x : xd, y : yd, is_training : False,
                    stop_the_gradient: epoch < refine_k}
    
    # For saving checkpoints
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Initialise all variables
        tf.initialize_all_variables().run(session=sess)
        
        for oppy in assign_ops:
            sess.run(oppy)
        
        # Merge all the summaries
        merged = tf.merge_all_summaries()
        if FLAGS.save_stuff:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '_k=' + str(FLAGS.refine_after_k) + '/train',
                                                  sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '_k=' + str(FLAGS.refine_after_k) + '/test')
        
        # Start training loops
        for epoch in range(0, FLAGS.max_steps):
            if epoch % 100 == 0:

              # Print accuracy and loss on test set
              summary, acc, loss_val = \
                  sess.run([merged, accuracy, loss], get_feed(cifar10, False, epoch))
                  
              if FLAGS.save_stuff:
                  test_writer.add_summary(summary, epoch)
              
              print ('\nEpoch', epoch, 
                     '\nTest accuracy:', acc, 
                     '\nTest loss    :', loss_val)
            
            if epoch % FLAGS.checkpoint_freq == 0:
              # Save model checkpoint
              if epoch > 0:
                  save_path = saver.save(sess, FLAGS.checkpoint_dir + \
                                         '/epoch'+ str(epoch) + '_k' + str(FLAGS.refine_after_k) + '.ckpt')
                  print("Model saved in file: %s" % save_path)
        
            # Do training update
            if FLAGS.save_stuff:
                summary, _ = sess.run([merged, train_op], 
                                      feed_dict=get_feed(cifar10, True, epoch))
                train_writer.add_summary(summary, epoch)
            else:
                sess.run([train_op], feed_dict=get_feed(cifar10, True, epoch))
        
        # Print the final accuracy
        summary, acc, loss_val = \
            sess.run([merged, accuracy, loss], get_feed(cifar10, False, epoch))
        
        if FLAGS.save_stuff:
            test_writer.add_summary(summary, epoch + 1)
        print ('\nFinal test accuracy:', acc,
               '\nFinal test loss    :', loss_val)
               
        save_path = saver.save(sess, FLAGS.checkpoint_dir + \
                               '/epoch'+ str(epoch + 1) + '_k' + str(FLAGS.refine_after_k) + '.ckpt')
        print("Model saved in file: %s" % save_path)
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir + '_k=' + str(FLAGS.refine_after_k)):
        tf.gfile.MakeDirs(FLAGS.log_dir + '_k=' + str(FLAGS.refine_after_k))

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
    train()

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
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')

    parser.add_argument('--save_stuff', type = bool, default = False,
                      help='Whether to save lots of logs')
    parser.add_argument('--fc_reg_str', type = float, default = 0.0,
                      help='Regularisation strength on fully-connected layers')
    parser.add_argument('--dropout_rate', type = float, default = 0.0,
                      help='Dropout rate for FC layers')
    parser.add_argument('--test_size', type = int, default = TEST_SIZE_DEFAULT,
                      help='Dropout rate for FC layers')
    parser.add_argument('--w_init', type = str, default = 'sqrt',
                      help='FC weights initialisation method')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()