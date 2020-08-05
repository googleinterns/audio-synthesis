# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains a collection of helpful utilitys
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras


class PadZeros2D(layers.Layer):
    """
    """
    
    def __init__(self, num_zeros=(1,1)):
        super(PadZeros2D, self).__init__()
        assert len(num_zeros) == 2
        
        self.shape_multiplier = np.array([1, num_zeros[0], num_zeros[1], 1], dtype=np.int32)
        self.num_zeros = num_zeros
        
        
    def call(self, x_in):
        output_shape = x_in.shape * self.shape_multiplier
        result = tf.nn.conv2d_transpose(
            x_in, tf.ones([1,1,1,x_in.shape[-1]]), output_shape, strides=self.num_zeros, padding='VALID'
        )
        return result
    

class Conv1DTranspose(layers.Layer): # pylint: disable=too-many-ancestors
    """Implementation of a 1-dimentional transpose convolution layer.

    This implementation is supose to emulate the interface of
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1DTranspose,
    so that it can be easily swapped out.

    NOTE: When this was implemented 'tf.keras.layers.Conv1DTranspose' was
          only available in the nightly build. Hence, this implementation using
          2D Transpose convolution.
    """

    def __init__(self, filters, kernel_size, strides, padding='same', **kwargs):
        super(Conv1DTranspose, self).__init__()
        self.conv_2d = layers.Conv2DTranspose(filters=filters,
                                       kernel_size=(kernel_size, 1),
                                       strides=(strides, 1),
                                       padding=padding,
                                       **kwargs)

    def call(self, x_in): # pylint: disable=arguments-differ
        x_in = tf.expand_dims(x_in, axis=2)
        x_up = self.conv_2d(x_in)
        x_up = tf.squeeze(x_up, axis=2)

        return x_up

#class HarmonicConvolution(keras.Layer):
 #   def __init__(self, filters, K, N, T)
    
class DeformableConvolutional2D(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(DeformableConvolutional2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        
        # Construct the receptive field
        self.R = [[]]
        
        
        self.offset_kernel = layers.Conv2D(filters=kernel_size * kernel_size, kernel_size=kernel_size, strides=1, padding='SAME')
        
    def call(self, x_in):
        x_shape = x_in.shape
        batch_size, in_w, in_h, channels_in = x_shape
        offsets = self.offset_kernel(x_in)
        print(offsets.shape)
        offsets = tf.reshape(offsets, (x_shape[0], x_shape[1], x_shape[2], self.kernel_size, self.kernel_size))
        
        # TODO: Include the bias.
        
        # TODO: Add the center point
        y, x = _get_conv_indices([in_h, in_w])
        print(y.shape)
        print(x.shape)
        
        
        # Have the final (floating point) indicies. #
        
        
        
        
    
class HarmonicConvolutionFilter(layers.Layer):
    def __init__(self, in_filters, out_filters, K, T):
        super(HarmonicConvolutionFilter, self).__init__()
        self.T = T
        self.K = K
        # First we construct the harmonic series
        #harmonic_series = []
        #for n in range(N):
        k_range = np.arange(1, K+1, 1, dtype=np.float32)
        self.series = k_range# * (1.0 / n)
        #harmonic_series.append(series)
        self.time = np.arange(-T, T+1, 1, dtype=np.int32)
        
        #self.filters = tf.Variable()
        
    def call(self, x_in):
        print(x_in.shape)
        # Pad edges of input so that we dont exceed the bounds
        # Currently, we just pad the time dimention and handle
        # the frequency dimention later.
        x_in_pad = tf.pad(x_in, [[0, 0], [self.T, self.T], [0,0], [0,0]])
        print(x_in_pad.shape)
        
        for tau in range(0, x_in.shape[1]):
            for omega in range(0, x_in.shape[2]):
                harmonic_series = tf.cast(self.series * omega, tf.int32)
                # Except here we need to handle the case where we are getting a fractional location
                print(harmonic_series)
                print(tau+self.time)
                print(-self.T + tau)
                print(self.T + tau+1)
                x_selection = x_in_pad[:, self.T + (-self.T + tau): self.T + (tau+self.T+1), :,: ]
                print(x_selection.shape)
                x_selection = tf.gather_nd(x_selection, tf.reshape(harmonic_series, (-1, 1, 1)))
                print(x_selection.shape)
                
                sys.exit(0)
                # Handle padding. Or prehapse we can do this before hand?
                #if x_selection.shape[2] < len(harmonic_series):
                #    x_selection = tf.pad(x_selection, [[0,0],[0,0],[0,len(harmonic_series) - x_selection.shape[2]],[0,0]])
                #print(x_selection.shape)
                
                # Multiply by filters
                
                
                # Insert into modified image
    
        