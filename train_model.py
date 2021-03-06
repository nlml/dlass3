from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
import cifar10_siamese_utils
from cifar10_siamese_utils import create_dataset
from convnet import ConvNet
from siamese import Siamese
from sklearn.manifold import TSNE
import cPickle

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 1000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

TEST_SIZE_DEFAULT = 1000
TRAIN_SIZE_LM_DEFAULT = 1000
SIAMESE_VALI_NTUPLES_DEFAULT = 50

CHECKPOINT_PATH_TO_LOAD_FROM_DEFAULT = 'checkpoints_siamesenew/'
CHECKPOINT_FILE_DEFAULT = 'epoch50000.ckpt'
SIAMESE_MARGIN_DEFAULT = 0.1
TRAIN_MODEL_DEFAULT = 'siamese'
IS_TRAIN_DEFAULT = False
SIAMESE_FRACTION_SAME_DEFAULT = 0.2

#### DELETE LATER #######
tf.reset_default_graph()
#########################

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
LOG_DIR_DEFAULT = './logs/test'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def wrap_cifar(cifar10, FLAGS):
    if FLAGS.standardise:
        class DataWrap():
            def __init__(self, dataset, sd):
                self.dataset = dataset
                self.sd = sd
                self.images = self.dataset.images / self.sd
                self.labels = self.dataset.labels
            def next_batch(self, sz, fraction_same=SIAMESE_FRACTION_SAME_DEFAULT):
                if FLAGS.train_model == 'siamese':
                    x, y = self.dataset.next_batch(sz)
                else:
                    x, y = self.dataset.next_batch(sz, fraction_same)
                return x / self.sd, y
        class CIFARWrapper():
            def __init__(self, cifar10_obj):
                self.sd = np.std(cifar10_obj.train.images, axis=0)
                self.train = DataWrap(cifar10_obj.train, self.sd)
                self.test = DataWrap(cifar10_obj.test, self.sd)
        cifar10 = CIFARWrapper(cifar10)
    return cifar10
    
def standard_cifar10_get(FLAGS):
    if FLAGS.train_model == 'siamese':
        cifar10     = cifar10_siamese_utils.get_cifar10(FLAGS.data_dir)
    else:
        cifar10     = cifar10_utils.get_cifar10(FLAGS.data_dir)
    cifar10         = wrap_cifar(cifar10, FLAGS)
    image_shape     = cifar10.train.images.shape[1:4]
    num_classes     = cifar10.test.labels.shape[1]
    return cifar10, image_shape, num_classes 

def train_step(loss, mini=True):
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
    if mini:
        train_op = optimizer.minimize(loss)
    else:
        train_op = optimizer.minimize(-loss)
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
    
    # Cifar10 stuff
    cifar10, image_shape, num_classes = standard_cifar10_get(FLAGS)
    
    # Placeholder variables
    x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
    
    # CNN model
    model = ConvNet(is_training=is_training, 
                    dropout_rate=FLAGS.dropout_rate, 
                    save_stuff=FLAGS.save_stuff, 
                    fc_reg_str=FLAGS.fc_reg_str)
    
    # Get logits, loss, accuracy, train optimisation step
    logits   = model.inference(x)
    accuracy = model.accuracy(logits, y)
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss     = model.loss(logits, y) + reg_loss
    tf.scalar_summary('loss_incl_reg', loss)
    train_op = train_step(loss)
    
    # Function for getting feed dicts
    def get_feed(c, train=True):
        if train:
            xd, yd = c.train.next_batch(FLAGS.batch_size)
            return {x : xd, y : yd, is_training : True}
        else:
            xd, yd = c.test.images[:FLAGS.test_size], c.test.labels[:FLAGS.test_size]
            return {x : xd, y : yd, is_training : False}
    
    # For saving checkpoints
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Initialise all variables
        tf.initialize_all_variables().run(session=sess)
        
        # Merge all the summaries
        merged = tf.merge_all_summaries()
        if FLAGS.save_stuff:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
                                                  sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
        
        # Start training loops
        for epoch in range(0, FLAGS.max_steps):
            if epoch % 100 == 0:

              # Print accuracy and loss on test set
              summary, acc, loss_val = \
                  sess.run([merged, accuracy, loss], get_feed(cifar10, False))
                  
              if FLAGS.save_stuff:
                  test_writer.add_summary(summary, epoch)
              
              print ('\nEpoch', epoch, 
                     '\nTest accuracy:', acc, 
                     '\nTest loss    :', loss_val)
            
            if epoch % FLAGS.checkpoint_freq == 0:
              # Save model checkpoint
              if epoch > 0:
                  save_path = saver.save(sess, FLAGS.checkpoint_dir + \
                                         '/epoch'+ str(epoch) + '.ckpt')
                  print("Model saved in file: %s" % save_path)
        
            # Do training update
            if FLAGS.save_stuff:
                summary, _ = sess.run([merged, train_op], 
                                      feed_dict=get_feed(cifar10, True))
                train_writer.add_summary(summary, epoch)
            else:
                sess.run([train_op], feed_dict=get_feed(cifar10, True))
        
        # Print the final accuracy
        summary, acc, loss_val = \
            sess.run([merged, accuracy, loss], get_feed(cifar10, False))
        
        if FLAGS.save_stuff:
            test_writer.add_summary(summary, epoch + 1)
        print ('\nFinal test accuracy:', acc,
               '\nFinal test loss    :', loss_val)
               
        save_path = saver.save(sess, FLAGS.checkpoint_dir + \
                               '/epoch'+ str(epoch + 1) + '.ckpt')
        print("Model saved in file: %s" % save_path)
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
    
    # Cifar10 stuff
    cifar10, image_shape, num_classes = standard_cifar10_get(FLAGS)
    
    # Placeholder variables
    x1 = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x1')
    x2 = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x2')
    y = tf.placeholder(tf.float32, shape=(None), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
    margin = tf.placeholder(tf.float32, shape=(), name='margin')
    
    # CNN model
    model = Siamese(is_training=is_training, 
                    dropout_rate=FLAGS.dropout_rate, 
                    save_stuff=FLAGS.save_stuff, 
                    fc_reg_str=FLAGS.fc_reg_str)
    
    # Get outputs of two siamese models, loss, train optimisation step
    l2_out_1        = model.inference(x1)
    l2_out_2        = model.inference(x2, reuse=True)
    loss_no_reg, d2 = model.loss(l2_out_1, l2_out_2, y, margin)
    reg_loss        = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_w_reg      = loss_no_reg + reg_loss
    accuracy        = model.accuracy(d2, y, margin)
    tf.scalar_summary('loss_incl_reg', loss_w_reg)
    train_op        = train_step(loss_w_reg)
    
    validation_tuples = create_dataset(cifar10.test, 
                                       num_tuples=FLAGS.siamese_vali_ntuples,
                                       batch_size=FLAGS.batch_size,
                                       fraction_same=FLAGS.siamese_fraction_same)
    xv1, xv2, yv = np.vstack([i[0] for i in validation_tuples]),\
                   np.vstack([i[1] for i in validation_tuples]),\
                   np.hstack([i[2] for i in validation_tuples])
    
    num_val_chunks = 10
    assert (FLAGS.siamese_vali_ntuples % num_val_chunks) == 0
    chunks = range(0, xv1.shape[0], int(xv1.shape[0] / num_val_chunks)) + \
             [int(FLAGS.siamese_vali_ntuples * FLAGS.batch_size)]
    
    # Function for getting feed dicts
    def get_feed(c, train=True, chunk=None, chunks=None):
        if train=='train' or train=='t':
            xd1, xd2, yd = \
                c.train.next_batch(FLAGS.batch_size, FLAGS.siamese_fraction_same)
            return {x1 : xd1, x2 : xd2, y : yd, is_training : True,
                    margin : FLAGS.siamese_margin}
        elif train=='vali' or train=='v' or train=='validation':
            if chunk is None:
                return {x1 : xv1, x2 : xv2, y : yv, is_training : False, 
                        margin : FLAGS.siamese_margin}     
            else:
                st, en = chunks[chunk], chunks[chunk+1]
                return {x1 : xv1[st:en], x2 : xv2[st:en], y : yv[st:en],
                        is_training : False, 
                        margin : FLAGS.siamese_margin} 
        else:
            pass
        # TODO Implement test set feed dict siamese
    
    # For saving checkpoints
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Initialise all variables
        tf.initialize_all_variables().run(session=sess)
        
        # Merge all the summaries
        merged = tf.merge_all_summaries()
        if FLAGS.save_stuff:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
                                                  sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
        
        # Start training loops
        for epoch in range(0, FLAGS.max_steps):
            if epoch % 100 == 0:
                
              # Print accuracy and loss on validation set
              accuracies = []
              losses = []
              for i in range(num_val_chunks):
                  loss_val, acc = \
                      sess.run([loss_no_reg, accuracy], 
                               get_feed(cifar10, 'vali', i, chunks))
                  accuracies.append(acc)
                  losses.append(loss_val)
                  
#              if FLAGS.save_stuff:
#                  test_writer.add_summary(summary, epoch)
              
              print ('\nEpoch', epoch, 
                     '\nValidation accuracy:', np.mean(accuracies), 
                     '\nValidation loss    :', np.mean(losses))

            if epoch % FLAGS.checkpoint_freq == 0:
              # Save model checkpoint
              if epoch > 0:
                  save_path = saver.save(sess, FLAGS.checkpoint_dir + \
                                         '/epoch'+ str(epoch) + '.ckpt')
                  print("Model saved in file: %s" % save_path)
        
            # Do training update
            if FLAGS.save_stuff:
                summary, _ = sess.run([merged, train_op], 
                                      feed_dict=get_feed(cifar10, 'train'))
                train_writer.add_summary(summary, epoch)
            else:
                sess.run([train_op], feed_dict=get_feed(cifar10, 'train'))
        
        # Print the final accuracy
        summary, loss_val = \
            sess.run([merged, loss_no_reg], get_feed(cifar10, 'vali'))
        
        if FLAGS.save_stuff:
            test_writer.add_summary(summary, epoch + 1)
        print ('\nFinal validation loss    :', loss_val)
    
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
    print('doing feature extraction...')
    
    tf.reset_default_graph()
    
    sess = tf.Session()
    
    cifar10, image_shape, num_classes = standard_cifar10_get(FLAGS)
    
    if FLAGS.train_model == 'siamese':
        # Construct siamese graph
    
        # Placeholder variables
        x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x1')
        y = tf.placeholder(tf.float32, shape=(None), name='y')
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
        
        # CNN model
        model = Siamese(is_training=is_training, 
                        dropout_rate=FLAGS.dropout_rate, 
                        save_stuff=FLAGS.save_stuff, 
                        fc_reg_str=FLAGS.fc_reg_str)
        
        # Get outputs of two siamese models, loss, train optimisation step
        l2  = model.inference(x)
        #fc2 = model.fc2
        fc2 = l2
    
    else:        
        # Construct linear convnet graph
        
        x = tf.placeholder(tf.float32, shape=[None] + list(image_shape), name='x')
        y = tf.placeholder(tf.int32, shape=(None, num_classes), name='y')
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
        
        model = ConvNet(is_training=is_training, dropout_rate=FLAGS.dropout_rate)
        
        _ = model.inference(x)
        
        fc2 = model.fc2
        
    # Initialise all variables
    tf.initialize_all_variables().run(session=sess)
    
    # Restore checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.ckpt_path + FLAGS.ckpt_file) 
    
    # Get testing data for feed dict
    x_data_test, y_data_test = \
        cifar10.test.images[:FLAGS.test_size], cifar10.test.labels[:FLAGS.test_size]
        
    # Get the test set features at flatten, fc1 and fc2 layers
    flatten_features_test, fc1_features_test, fc2_features_test = \
        sess.run([model.flatten, model.fc1, fc2], 
                 {x : x_data_test, y : y_data_test, is_training : False})
    
    # Get t-SNE manifold of these features
    tsne = TSNE()
    manifold = tsne.fit_transform(fc2_features_test)
    
    # Save to disk for plotting later
    indices = np.arange(FLAGS.test_size)
    cPickle.dump((manifold, indices), open('manifold' + FLAGS.train_model + '.dump', 'wb'))
    
    # Get training data for feed dict
    x_data_train, y_data_train = \
        cifar10.train.images[:FLAGS.train_size_lm], cifar10.train.labels[:FLAGS.train_size_lm]
        
    # Get train set features at flatten, fc1 and fc2 layers
    flatten_features_train, fc1_features_train, fc2_features_train = \
        sess.run([model.flatten, model.fc1, fc2], 
                 {x : x_data_train, y : y_data_train, is_training : False})
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    features_list = [['flat', flatten_features_train, flatten_features_train],
                     ['fc1 ', fc1_features_train, fc1_features_test],
                     ['fc2 ', fc2_features_train, fc2_features_test]]
    for (name, features_train, features_test) in features_list:
        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(features_train, y_data_train)
        lm_test_predictions = classif.predict(features_test)
        acc = np.mean(np.argmax(y_data_test, 1)==np.argmax(lm_test_predictions, 1))
        print (name, 'accuracy =', np.round(acc*100, 2), '%')
    
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
    parser.add_argument('--train_model', type = str, default = TRAIN_MODEL_DEFAULT,
                      help='Type of model. Possible options: linear and siamese')
    parser.add_argument('--fc_reg_str', type = float, default = 0.0,
                      help='Regularisation strength on fully-connected layers')
    parser.add_argument('--dropout_rate', type = float, default = 0.0,
                      help='Dropout rate for FC layers')
    parser.add_argument('--siamese_margin', type = float, default = SIAMESE_MARGIN_DEFAULT,
                      help='Margin for siamese contrastive loss')
    parser.add_argument('--siamese_fraction_same', type = float, default = SIAMESE_FRACTION_SAME_DEFAULT,
                      help='Siamese sampling fraction_same')
    parser.add_argument('--ckpt_path', type = str, default = CHECKPOINT_PATH_TO_LOAD_FROM_DEFAULT,
                      help='Checkpoint path to load from (end with /)')
    parser.add_argument('--ckpt_file', type = str, default = CHECKPOINT_FILE_DEFAULT,
                      help='Checkpoint file name')
    parser.add_argument('--test_size', type = int, default = TEST_SIZE_DEFAULT,
                      help='Dropout rate for FC layers')
    parser.add_argument('--train_size_lm', type = int, default = TRAIN_SIZE_LM_DEFAULT,
                      help='Dropout rate for FC layers')
    parser.add_argument('--siamese_vali_ntuples', type = int, default = SIAMESE_VALI_NTUPLES_DEFAULT,
                      help='Dropout rate for FC layers')
    parser.add_argument('--is_train', type = bool, default = IS_TRAIN_DEFAULT,
                      help='Training or feature extraction')
    parser.add_argument('--save_stuff', type = bool, default = False,
                      help='Whether to save lots of logs')
    parser.add_argument('--standardise', type = bool, default = False,
                      help='Whether to divide images by stdev')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()