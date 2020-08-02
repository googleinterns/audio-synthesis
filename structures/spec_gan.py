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

"""An implemementation of the Generator and Discriminator for SpecGAN.

This file contains an implementation of the generator and discriminator
components for SpecGAN [https://arxiv.org/abs/1802.04208].
The official implementation of SpecGAN can be found online
(https://github.com/chrisdonahue/wavegan). The key difference between this implementation
and the offical one is that we use a larger kenel size to keep the model balanced
with our implementation of WaveGAN. We choose a kernel size of 6x6 (instead of 5x5),
such that it is divisible by the stride.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers
from tensorflow import keras

class Generator(keras.Model):
    """Implementation of the SpecGAN Generator Function."""

    def __init__(self, channels=1, activation=activations.linear, in_shape=(4, 4, 1024)):
        """Initilizes the SpecGAN Generator function.

        Paramaters:
            channels: The number of output channels.
                For example, for SpecGAN there is one
                output channel, and for SpecPhaseGAN there
                are two output channels.
            acitvation: Activation function applied to generation
                before being returned. Default is linear.
            in_shape: Transformed noise shape as input to the
                generator function.
        """

        super(Generator, self).__init__()

        self.activation = activation

        sequential = []
        sequential.append(layers.Dense(np.prod(in_shape)))
        sequential.append(layers.Reshape((in_shape)))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2DTranspose(filters=512, kernel_size=(6, 6),
                                                 strides=(2, 2), padding='same'))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2DTranspose(filters=256, kernel_size=(6, 6),
                                                 strides=(2, 2), padding='same'))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2DTranspose(filters=128, kernel_size=(6, 6),
                                                 strides=(2, 2), padding='same'))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2DTranspose(filters=64, kernel_size=(6, 6),
                                                 strides=(2, 2), padding='same'))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2DTranspose(filters=channels, kernel_size=(6, 6),
                                                 strides=(2, 2), padding='same'))

        self.l = keras.Sequential(sequential)

    def call(self, z_in):
        """Generates spectograms from input noise vectors.

        Args:
            z_in: A batch of random noise vectors. Expected shape
            is (batch_size, z_dim).

        Returns:
            The output from the generator network. Same number of
            batch elements.
        """

        return self.activation(self.l(z_in))

class Discriminator(keras.Model):
    """Implementation of the SpecGAN Discriminator Function."""

    def __init__(self, input_shape, weighting=1.0):
        """Initilizes the SpecGAN Discriminator function
        
        Args:
            input_shape: The required shape for inputs to the
                discriminator functions.
            weighting: The relative weighting of this discriminator in
                the overall loss.
        """
        
        super(Discriminator, self).__init__()

        self.in_shape = input_shape
        self.weighting = weighting
        
        sequential = []
        sequential.append(layers.Conv2D(filters=64, kernel_size=(6, 6),
                                        strides=(2, 2), padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(filters=128, kernel_size=(6, 6),
                                        strides=(2, 2), padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(filters=256, kernel_size=(6, 6),
                                        strides=(2, 2), padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(filters=512, kernel_size=(6, 6),
                                        strides=(2, 2), padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(filters=1024, kernel_size=(6, 6),
                                        strides=(2, 2), padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Flatten())
        sequential.append(layers.Dense(1))

        self.l = keras.Sequential(sequential)

    def call(self, x_in):
        """Produces discriminator scores for the inputs.

        Args:
            x_in: A batch of input data. Expected shape
            is expected to be consistant with self.in_shape.

        Returns:
            A batch of real valued scores. This is inlign with
            the WGAN setup.
        """

        x_in = tf.reshape(x_in, self.in_shape)
        return self.l(x_in)
