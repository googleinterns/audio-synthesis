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

"""An implemementation of the Generator and Discriminator for the
MIDI Conditional SpecGAN.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers
from tensorflow import keras
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """Implementation of the SpecGAN Generator Function."""

    def __init__(self, channels=1, activation=activations.linear, z_in_shape=(128, 1)):
        """Initilizes the SpecGAN Generator function.

        Args:
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

        # Pre-process the random noise input.
        z_pre_process = []
        z_pre_process.append(layers.Dense(np.prod(z_in_shape)))
        z_pre_process.append(layers.Reshape(z_in_shape))
        z_pre_process.append(layer_utils.Conv1DTranspose(
            filters=512, strides=4, kernel_size=8
        ))
        self.z_pre_process = keras.Sequential(z_pre_process)

        # Pre-processing stack for the conditioning information.
        c_pre_process_1x1 = []
        c_pre_process_1x1.append(layers.Conv1D(
            filters=512, strides=1, kernel_size=1, padding='same'
        ))
        self.c_pre_process_1x1 = keras.Sequential(c_pre_process_1x1)
        
        sequential = []
        sequential.append(layers.Conv2D(
            filters=512, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2D(
            filters=256, kernel_size=(6, 6), strides=(2, 1), padding='same'
        ))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2D(
            filters=128, kernel_size=(6, 6), strides=(1, 1), padding='same'
        ))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2D(
            filters=64, kernel_size=(6, 6), strides=(1, 1), padding='same'
        ))
        sequential.append(layers.ReLU())
        sequential.append(layers.Conv2D(
            filters=channels, kernel_size=(6, 6), strides=(1, 1), padding='same'
        ))

        self.l = keras.Sequential(sequential)

    def call(self, z_in, c_in):
        """Generates spectograms from input noise vectors.

        Args:
            z_in: A batch of random noise vectors. Expected shape
                is (batch_size, z_dim).
            c_in: A batch of midi conditioning. Expected shape is
                (batch_size, num_states, 89), where 89 represents the 88
                piano keys plus the sustain pedal. 

        Returns:
            The output from the generator network. Same number of
            batch elements.
        """
        
        z_pre_processed = self.z_pre_process(z_in)
        c_pre_processed = self.c_pre_process_1x1(c_in)
        z_pre_processed = tf.expand_dims(z_pre_processed, axis=-1)
        c_pre_processed = tf.expand_dims(c_pre_processed, axis=-1)
        zc_in = tf.concat([z_pre_processed, c_pre_processed], axis=-1)
        
        output = self.activation(self.l(zc_in))
        return output

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

        # Pre-processing stack for the conditioning information.
        c_pre_process = []
        c_pre_process.append(layers.Conv1D(
            128, kernel_size=36, strides=2, padding='same'
        ))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(
            256, kernel_size=36, strides=2, padding='same'
        ))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        self.c_pre_process = keras.Sequential(c_pre_process)

        sequential = []
        sequential.append(layers.Conv2D(
            filters=64, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(
            filters=128, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(
            filters=256, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(
            filters=512, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv2D(
            filters=512, kernel_size=(6, 6), strides=(2, 2), padding='same'
        ))
        #sequential.append(layers.Conv2D(
        #    filters=3, kernel_size=(6, 6), strides=(1, 1), padding='same'
        #))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Flatten())
        sequential.append(layers.Dense(1))

        self.l = keras.Sequential(sequential)

    def call(self, x_in, c_in):
        """Produces discriminator scores for the inputs.

        Args:
            x_in: A batch of input data. Expected shape
                is expected to be consistant with self.in_shape.
            c_in: A batch of midi conditioning. Expected shape is
                (batch_size, num_states, 89), where 89 represents the 88
                piano keys plus the sustain pedal. 

        Returns:
            A batch of real valued scores. This is inlign with
            the WGAN setup.
        """

        c_pre_processed = self.c_pre_process(c_in)
        c_pre_processed = tf.expand_dims(c_pre_processed, axis=-1)
        x_in = tf.reshape(x_in, self.in_shape)

        xc_in = tf.concat([c_pre_processed, x_in], axis=-1)

        return self.l(xc_in)
