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
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import Model


class Conv1DTranspose(Model): # pylint: disable=too-many-ancestors
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
        self.conv_2d = Conv2DTranspose(filters=filters,
                                       kernel_size=(kernel_size, 1),
                                       strides=(strides, 1),
                                       padding=padding,
                                       **kwargs)

    def call(self, x_in): # pylint: disable=arguments-differ
        x_in = tf.expand_dims(x_in, axis=2)
        x_up = self.conv_2d(x_in)
        x_up = tf.squeeze(x_up, axis=2)

        return x_up
