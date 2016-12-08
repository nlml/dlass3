from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import regularizers


class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10, is_training=True, dropout_rate=0.,
                 save_stuff=False):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.save_stuff = save_stuff

        
    def _list_or_int(self, inp, pad_with_ones=False):
        
        # For getting kernel sizes from a single number or a length-2 list
        if type(inp) is list:
            output = [1, inp[0], inp[1], 1]
        else:
            output = [1, inp, inp, 1]

        if pad_with_ones:
            return output
        else:
            return [output[1], output[2]]
        
    def _conv2d_layer(self, x, n, kernel_size, stride, inp_depth, out_depth, 
                      act_fn):
        '''
        x            : tensor
        n            : name; string
        kernel_size  : single int (symmetric) or list of length 2; filter sizes
        stride       : single int (symmetric) or list of length 2; stride sizes
        inp_depth    : int
        out_depth    : int
        act_fn       : e.g. tf.nn.relu
        '''
        
        # Naming scope
        with tf.variable_scope(n):
            
            # Kernel size
            kernel  = self._list_or_int(kernel_size, pad_with_ones=False)
            
            # Weights
            W_shape = [kernel[0], kernel[1], inp_depth, out_depth]
            
            sd      = 1./np.sqrt((int(np.product(x.get_shape()[1:]))))
            W_init  = tf.truncated_normal_initializer(stddev=sd, 
                                                      dtype=tf.float32)
            W       = tf.get_variable('W', W_shape, initializer=W_init)
            
            # Biases
            b_init  = tf.constant_initializer(0)
            b       = tf.get_variable('b', out_depth, initializer=b_init)
            
            # Strides
            strides  = self._list_or_int(stride, pad_with_ones=True)
            
            # Convolve
            pre_act = b + tf.nn.conv2d(x, W, padding='SAME', strides=strides)
            self._histsum('pre_act', pre_act)
            
            # Activate
            if act_fn is not None:
                output = act_fn(pre_act)
            else:
                output = pre_act
                
            self._histsum('pre_pool', output)
            
        self._get_weights_and_bias_summaries(n)
            
        return output
        
        
    def _pool_layer(self, x, n, pool_size, pool_stride, 
                    pool_type=tf.nn.max_pool):
        '''
        x           : tensor
        n           : name; string
        pool_size   : int (symmetric) or list of 2 ints; size of pooling op
        pool_stride : int (symmetric) or list of 2 ints; stride of pooling op
        pool_type   : e.g. tf.nn.max_pool
        '''
        
        # Naming scope
        with tf.variable_scope(n):
            
            # Kernel size
            kernel = self._list_or_int(pool_size, pad_with_ones=True)
            
            # Strides
            stride = self._list_or_int(pool_stride, pad_with_ones=True)
            
            # Apply pooling
            out = pool_type(x, ksize=kernel, strides=stride, padding='SAME')
            self._histsum('post_pool', out)
        
        return out
        
    def _fc_layer(self, x, n, output_shape=None, act_fn=None, 
                      reg_strength=0., input_shape=None):
        '''
        x            : tensor
        n            : name; string
        input_shape  : int
        output_shape : int
        act_fn       : e.g. tf.nn.relu
        reg_strength : float - regularisation strength
        '''
        
        # Naming scope
        with tf.variable_scope(n):
            
            # Infer input shape from x
            if input_shape is not None:
                if x.get_shape()[1] != input_shape:
                    raise Warning('x shape in hidden layer != given shape. ' +\
                                  'This may cause problems...')
            else:
                input_shape = x.get_shape()[1]
            
            # Weights
            W_shape = [input_shape, output_shape]
            
            sd      = 1./np.sqrt(int(np.product(x.get_shape()[1:])))
            W_init  = tf.truncated_normal_initializer(stddev=sd, 
                                                      dtype=tf.float32)
            W_reg   = regularizers.l2_regularizer(reg_strength)
            W       = tf.get_variable('W', W_shape, initializer=W_init, 
                                      regularizer=W_reg)
            
            # Biases
            b_init  = tf.constant_initializer(0)
            b       = tf.get_variable('b', output_shape, initializer=b_init)
            
            # Linear transform
            pre_act = tf.matmul(x, W) + b
            self._histsum('pre_act', pre_act)
            
            # Activate
            if act_fn is not None:
                pre_drop = act_fn(pre_act)
            else:
                pre_drop = pre_act
                
            self._histsum('pre_drop', pre_drop)
            
            # Dropout
            output = tf.cond(self.is_training,
                             lambda: tf.nn.dropout(pre_drop, 
                                                   1.0 - self.dropout_rate),
                             lambda: pre_drop)
            
            self._histsum('post_drop', output)
            
        self._get_weights_and_bias_summaries(n)
                             
        return output
        
    def _histsum(self, n, x):
        if self.save_stuff:
            tf.histogram_summary(tf.get_variable_scope().name + '/' + n, x)
        
    def _histmean_summary(self, varname, scope):
        if self.save_stuff:
            var = tf.get_variable(varname)
            tf.histogram_summary(scope + '/' + varname + '_hist', var)
      
    def _get_weights_and_bias_summaries(self, scope):
        if self.save_stuff:
            with tf.variable_scope(scope, reuse=True):
                # Get variables
                self._histmean_summary('W', scope)
                self._histmean_summary('b', scope)
        
    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
        # can probably infer inp_depth from x later...
            self.conv1 = self._conv2d_layer(x, 'conv1',
                                            kernel_size = 5, 
                                            stride      = 1, 
                                            inp_depth   = 3,
                                            out_depth   = 64,
                                            act_fn      = tf.nn.relu)
            self.pool1 = self._pool_layer(self.conv1, 'pool1', pool_size=3, pool_stride=2)
            
            self.conv2 = self._conv2d_layer(self.pool1, 'conv2',
                                            kernel_size = 5, 
                                            stride      = 1, 
                                            inp_depth   = 64,
                                            out_depth   = 64,
                                            act_fn      = tf.nn.relu)
            self.pool2 = self._pool_layer(self.conv2, 'pool2', pool_size=3, pool_stride=2)
            
            with tf.variable_scope('flatten'):
                self.flatten = tf.reshape(self.pool2, [-1, 64 * 64])
                
            self.fc1 = self._fc_layer(self.flatten, 'fc1',
                                            act_fn       = tf.nn.relu,
                                            reg_strength = 0.001,
                                            output_shape = 384)
            self.fc2 = self._fc_layer(self.fc1, 'fc2',
                                            act_fn       = tf.nn.relu,
                                            reg_strength = 0.001,
                                            output_shape = 192)
            self.logits = self._fc_layer(self.fc2, 'logits',
                                             act_fn       = None,
                                             reg_strength = 0.001,
                                             output_shape = 10)
                
            ########################
            # END OF YOUR CODE    #
            ########################
        return self.logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        correct_prediction = tf.equal(tf.argmax(labels, 1),
                                      tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            logits, labels, name='crossentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.scalar_summary('crossentropyloss', loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
