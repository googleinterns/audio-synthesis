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
from tensorflow.keras import layers, activations
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
    
    
class AbstractDeformableConvolutional2D(layers.Layer):
    def __init__(self, filters, kernel_size, input_filters, activation=activations.linear):
        super(AbstractDeformableConvolutional2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_filters = input_filters
        self.activation = activation
        
        # Construct the receptive field
        self.R = np.reshape([np.arange(-kernel_size[0] // 2 + 1, kernel_size[0]//2 + 1, 1) for i in range(kernel_size[1])], (-1)).astype(np.float32)
        self.R = np.reshape(self.R, (1, 1, 1, -1))
        
        self.kernel_shape = self.kernel_size + (input_filters, filters)
        self.kernel = tf.Variable(np.random.normal(size=self.kernel_shape, scale=0.01).astype(np.float32), trainable=True)
        
    def _get_offset(self, x_in):
        raise NotImplementedError()
        
    def call(self, x_in):
        x_shape = x_in.shape
        batch_size, in_h, in_w, channel_in = x_shape
        filter_h, filter_w = self.kernel_size
        
        y_offset, x_offset = self._get_offset(x_in)
        
        x_origin, y_origin = tf.meshgrid(tf.range(x_shape[2]), tf.range(x_shape[1]))
        x_origin = tf.cast(tf.reshape(x_origin, (1, x_shape[1], x_shape[2], 1)), tf.float32) + 1.0
        y_origin = tf.cast(tf.reshape(y_origin, (1, x_shape[1], x_shape[2], 1)), tf.float32) + 1.0
        
        x_relative = x_origin + x_offset + self.R
        y_relative = y_origin + x_offset + self.R

        # Rather than add lots of padding, add border of zeros
        # and then clip to border. Hence, any value that sits outside
        # the image bounds is clipped to a position with a zero.
        x_in_padded = tf.pad(x_in, [[0,0],[1,1], [1,1], [0,0]])
        x_in_padded = tf.cast(x_in_padded, tf.float32)
        #x_relative = tf.clip_by_value(x_relative, 0, in_w)
        #y_relative = tf.clip_by_value(y_relative, 0, in_h)
        # Have the final (floating point) indicies. #
        
        # Get coordinates of the points around (x,y)
        x0 = tf.cast(tf.floor(x_relative), tf.int32)
        x1 = x0 + 1
        
        y0 = tf.cast(tf.floor(y_relative), tf.int32)
        y1 = y0 + 1
        
        x0 = tf.clip_by_value(x0, 0, in_w+1)
        x1 = tf.clip_by_value(x1, 0, in_w+1)
        y0 = tf.clip_by_value(y0, 0, in_h+1)
        y1 = tf.clip_by_value(y1, 0, in_h+1)
                
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [_get_pixel_values_at_point(x_in_padded, i) for i in indices]

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        
        # weights
        w0 = (y1 - y_relative) * (x1 - x_relative)
        w1 = (y1 - y_relative) * (x_relative - x0)
        w2 = (y_relative - y0) * (x1 - x_relative)
        w3 = (y_relative - y0) * (x_relative - x0)
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        
        pixles = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
        
        pixles = tf.reshape(pixles, [batch_size, in_h, in_w, filter_h, filter_w, channel_in])
        pixles = tf.transpose(pixles, [0, 1, 3, 2, 4, 5])
        pixles = tf.reshape(pixles, [batch_size, in_h * filter_h, in_w * filter_w, channel_in])

        out = tf.nn.conv2d(pixles, self.kernel, strides=[1, filter_h, filter_w, 1], padding='VALID')
        
        return self.activation(out)
        
def _get_pixel_values_at_point(inputs, indices):
    """get pixel values
    :param inputs:
    :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
    :return:
    """
    
    y, x = indices
    batch, h, w, n = y.shape

    batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
    b = tf.tile(batch_idx, (1, h, w, n))
    pixel_idx = tf.stack([b, y, x], axis=-1)

    result = tf.gather_nd(inputs, pixel_idx)
    return result
    
class DeformableConvolutional2D(AbstractDeformableConvolutional2D):
    def __init__(self, filters, kernel_size, input_filters, strides=(1, 1), padding='same', activation=activations.linear):
        super(DeformableConvolutional2D, self).__init__(filters, kernel_size, input_filters, activation)
        
        self.strides = strides
        self.padding = padding
        
        self.offset_kernel = layers.Conv2D(
            filters=kernel_size[0] * kernel_size[1] * 2,
            kernel_size=kernel_size, strides=self.strides,
            padding=self.padding
        )
        self.offset_bias = tf.Variable(
            np.random.normal(
                size=(1, 1, 1, kernel_size[0] * kernel_size[1] * 2), scale=0.01
            ).astype(np.float32),
            trainable=True
        )
        
    def _get_offset(self, x_in):
        x_shape = x_in.shape
        offsets = self.offset_kernel(x_in)
        offsets = offsets + self.offset_bias
        
        offsets = tf.reshape(offsets, (x_shape[0], x_shape[1], x_shape[2], -1, 2))
        y_offset = offsets[:,:,:,:,0]
        x_offset = offsets[:,:,:,:,1]
        
        return y_offset, x_offset
    

class HarmonicConvolutionFilter(AbstractDeformableConvolutional2D):
    def __init__(self, harmonic_series, time, anchor, filters, input_filters, activation=activations.linear):
        super(HarmonicConvolutionFilter, self).__init__(filters, (harmonic_series, 2*time),  input_filters, activation)
        self.T = T
        self.K = K
        k_range = np.arange(1, harmonic_series+1, 1, dtype=np.float32)
        self.series = k_range * (1.0 / anchor)
        #harmonic_series.append(series)
        self.time = np.arange(-time, time+1, 1, dtype=np.int32)
        
        self.x_offset = tf.reshape(self.time, (1, 1, 1, -1))
        self.y_offset = tf.reshape(self.series, (1, 1, 1, -1))
        
    def _get_offset(self, x_in):
        return self.x_offset, self.y_offset