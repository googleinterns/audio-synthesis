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
from tensorflow.keras import layers
from tensorflow import keras


class Conv1DTranspose(keras.Layer): # pylint: disable=too-many-ancestors
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

    
class HarmonicConvolution(keras.Layer):
    def __init__(self, filters, K, N, T)
    
    
    
class HarmonicConvolutionFilter_n():
    def __init__(self, num_filters, K, n, T):
        
        self.T = T
        self.K = K
        # First we construct the harmonic series
        #harmonic_series = []
        #for n in range(N):
        k_range = np.arange(1, K+1, 1, dtype=np.float32)
        self.series = k_range * (1.0 / n)
        #harmonic_series.append(series)
        self.time = np.arange(-T, T+1, 1, dtype=np.float32)
        
        self.filters = tf.Variable()
        
    def call(self, x_in):
        # Pad edges of input so that we dont exceed the bounds
        x_in_pad = tf.pad(x_in, [[0, 0], [T]])
        
        for tau in range(0, x_in.shape[1]):
            for omega in range(0, x_in.shape[2]):
                # Except here we need to handle the case where we are getting a fractional location
                x_selection = x_in[:, tau + self.time, omega + self.series, : ]
                
                # Handle padding. Or prehapse we can do this before hand?
                
                # Multiply by filters
                
                # Insert into modified image
    
        