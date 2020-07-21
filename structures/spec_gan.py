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

from tensorflow.keras import activations, layers
from tensorflow import keras

class Generator(keras.Model):
    """Implementation of the SpecGAN Generator Function."""

    def __init__(self, channels=1, activation=activations.linear):
        """Initilizes the SpecGAN Generator function.

        Paramaters:
            channels: The number of output channels.
                For example, for SpecGAN there is one
                output channel, and for SpecPhaseGAN there
                are two output channels.
            acitvation: Activation function applied to generation
                before being returned. Default is linear.
        """

        super(Generator, self).__init__()

        self.activation = activation

        sequential = []
        sequential.append(layers.Dense(4 * 4 * 1024))
        sequential.append(layers.Reshape((4, 4, 1024)))
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
        return self.activation(self.l(z_in))

class Discriminator(keras.Model):
    """Implementation of the SpecGAN Discriminator Function."""

    def __init__(self, input_shape, weighting=1.0):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
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
        x_in = tf.reshape(x_in, self.input_shape)
        return self.l(x_in)
