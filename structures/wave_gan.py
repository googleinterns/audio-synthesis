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

"""An implemementation of the Generator and Discriminator for WaveGAN.

This file contains an implementation of the generator and discriminator
components for the WaveGAN [https://arxiv.org/abs/1802.04208] model.
The official implementation of WaveGAN can be found online
(https://github.com/chrisdonahue/wavegan) and contains a more general
implementation of WaveGAN. The key difference between this implementation
and the offical one is that we do not use Phase Shuffle to avoid checkerboarding,
instead we choose a kernel size of 36 (instead of 25),
such that it is divisible by the stride.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """The Generator function for WaveGAN.

    The model takes a latent vector as input and transforms it into
    a signal with 16 time-steps and 1024 channels. Five transpose
    convolution layers upscale in the time dimention to 16**14 samples
    """

    def __init__(self, name='generator'):
        super(Generator, self).__init__()
        sequential = []
        sequential.append(layers.Dense(16 * 1024))
        sequential.append(layers.Reshape((16, 1024)))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=512, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=256, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=128, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=64, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=1, strides=4, kernel_size=36))

        self.l = keras.Sequential(sequential, name=name)

    def call(self, z_in):
        output = self.l(z_in)
        return output


class Discriminator(keras.Model):
    """Implementation of the discriminator for WaveGAN

    The model takes as input a real or fake waveform and,
    following the WGAN framework, produces a real valued
    output.
    """

    def __init__(self, input_shape, weighting=1.0, name='discriminator'):
        super(Discriminator, self).__init__()
        
        self.in_shape = input_shape
        self.weighting = weighting
        
        sequential = []
        sequential.append(layers.Conv1D(64, kernel_size=36, strides=4))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(128, kernel_size=36, strides=4))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(256, kernel_size=36, strides=4))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(512, kernel_size=36, strides=4))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(1024, kernel_size=36, strides=4))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Flatten())
        sequential.append(layers.Dense(1))

        self.l = keras.Sequential(sequential, name=name)

    def call(self, x_in):
        x_in = tf.reshape(x_in, self.in_shape)
        output = self.l(x_in)
        return output
