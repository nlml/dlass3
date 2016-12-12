from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import regularizers
from convnet import ConvNet

class VGGReadOut(ConvNet):
    '''
    We just override the inference method from ConvNet
    '''
        
    def inference(self, pool5, w_init):
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
        with tf.variable_scope('VGGReadOut'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
        # can probably infer inp_depth from x later...
            with tf.variable_scope('flatten'):
                self.flatten = tf.reshape(pool5, [-1, 512])
                
            self.fc1 = self._fc_layer(self.flatten, 'fc1',
                                            act_fn       = tf.nn.relu,
                                            reg_strength = self.fc_reg_str,
                                            output_shape = 384,
                                            init         = w_init)
            self.fc2 = self._fc_layer(self.fc1, 'fc2',
                                            act_fn       = tf.nn.relu,
                                            reg_strength = self.fc_reg_str,
                                            output_shape = 192,
                                            init         = w_init)
            self.logits = self._fc_layer(self.fc2, 'logits',
                                             act_fn       = None,
                                             reg_strength = self.fc_reg_str,
                                             output_shape = 10,
                                             init         = w_init)
                
            ########################
            # END OF YOUR CODE    #
            ########################
        return self.logits
