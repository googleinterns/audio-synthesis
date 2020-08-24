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

"""Network descriptions for the generator and discriminators
of the Conditional WaveSpecGAN. Designed for MIDI conditioning.
Conditioning is expected to be a matrix of size (1024, 89), where
the first 512 state vectors are for the previous second of audio and
the next 512 state vectors are for the current second.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """The Generator function for Conditional WaveSpecGAN.
    """

    def __init__(self, z_in_shape=(128, 256)):
        """Initilizer for the Conditional WaveSpecGAN generator
        function.

        Args:
            z_in_shape: Desired shape of the z input after the
                initial dense layer.
        """

        super(Generator, self).__init__()
        
        # Pre-process the random noise input. Input shape is
        # [-1, batch_size] and the output shape is [-1, 1024, 1024]
        z_pre_process = []
        z_pre_process.append(layers.Dense(np.prod(z_in_shape)))
        z_pre_process.append(layers.Reshape(z_in_shape))
        z_pre_process.append(layer_utils.Conv1DTranspose(
            filters=1024, strides=8, kernel_size=36
        ))
        z_pre_process.append(layers.ReLU())
        self.z_pre_process = keras.Sequential(z_pre_process)

        # Pre-processing stack for the conditioning information, input is shape
        # [-1, 1024, 89] output is [-1, 1024, 1024].
        c_pre_process = []
        c_pre_process.append(layers.Conv1D(
            filters=1024, strides=1, kernel_size=36, padding='same'
        ))
        c_pre_process.append(layers.ReLU())
        self.c_pre_process = keras.Sequential(c_pre_process)

        # Concatenate processed conditioning and spectrogram in the last
        # channel, giving an input of shape [-1, 1024, 2048].
        sequential = []
        sequential.append(layer_utils.Conv1DTranspose(filters=512, strides=1, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=256, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=128, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=64, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=1, strides=2, kernel_size=36))

        self.l = keras.Sequential(sequential)

    def call(self, z_in, c_in):
        """Generates audio from the given conditioning and
        random noise.

        Args:
            c_in: The conditioning information. Expected shape is
                [batch_size, 1024, 89]
            z_in: The random noise. Expected shape is
                [batch_size, z_dim]

        Returns:
            The generated audio. The shape is
            [batch_size, signal_length]
        """

        z_pre_processed = self.z_pre_process(z_in)
        c_pre_processed = self.c_pre_process(c_in)
        zc_in = z_pre_processed + c_pre_processed
        output = self.l(zc_in)
        return output

class WaveformDiscriminator(keras.Model):
    """Implementation of the waveform discriminator for
    Conditional WaveSpecGAN.
    """

    def __init__(self, input_shape, weighting=1.0):
        """Initilizer for the waveform discriminator.

        Args:
            input_shape: The required shape for data at input to
                the discriminator.
            weighting: The relative weighting of the discriminator in
                the final loss.
        """

        super(WaveformDiscriminator, self).__init__()

        self.in_shape = input_shape
        self.weighting = weighting

        # Pre-processing stack for the conditioning information, input is shape
        # [-1, 1024, 89] output is [-1, 128, 256].
        c_pre_process = []
        c_pre_process.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(256, kernel_size=36, strides=2, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        self.c_pre_process = keras.Sequential(c_pre_process)

        # Pre-processing stack for the input data. Input is 
        # [-1, 2**14] output is [-1, 128, 256]
        x_pre_process = []
        x_pre_process.append(layers.Conv1D(64, kernel_size=36, strides=4, padding='same'))
        x_pre_process.append(layers.LeakyReLU(alpha=0.2))
        x_pre_process.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        x_pre_process.append(layers.LeakyReLU(alpha=0.2))
        x_pre_process.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        x_pre_process.append(layers.LeakyReLU(alpha=0.2))
        x_pre_process.append(layers.Conv1D(256, kernel_size=36, strides=2, padding='same'))
        x_pre_process.append(layers.LeakyReLU(alpha=0.2))
        self.x_pre_process = keras.Sequential(x_pre_process)

        # Pre-processed x_in and c_in are concatenated to give a input
        # of shape [-1, 128, 512]
        sequential = []
        sequential.append(layers.Conv1D(512, kernel_size=36, strides=2, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(512, kernel_size=36, strides=2, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(1024, kernel_size=36, strides=2, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Flatten())
        sequential.append(layers.Dense(1))

        self.sequential = keras.Sequential(sequential)

    def call(self, x_in, c_in):
        """Produces a critic score for x_in given the conditioning
        information c_in.

        Args:
            x_in: A batch of real or generated data. Expected shape is
                [batch_size, signal_length]
            c_in: The asociated conditioning information. Expected shape
                is [batch_size, 1024, 89].

        Returns:
            The critic scores. Shape is [batch_size, 1]
        """

        x_in = tf.reshape(x_in, self.in_shape)

        x_processed = self.x_pre_process(x_in)
        c_processed = self.c_pre_process(c_in)
        xc_in = tf.concat([x_processed, c_processed], axis=-1)

        output = self.sequential(xc_in)
        return output

class SpectogramDiscriminator(keras.Model):
    """Implementation of the magnitude spectrum discriminator for
    Conditional WaveSpecGAN.
    """

    def __init__(self, input_shape, weighting=1.0):
        """Initilizer for the magnitude spectrum discriminator.

        Args:
            input_shape: The required shape for data at input to
                the discriminator.
            weighting: The relative weighting of the discriminator in
                the final loss.
        """

        super(SpectogramDiscriminator, self).__init__()

        self.in_shape = input_shape
        self.weighting = weighting

        # Pre-processing stack for the conditioning information, input is shape
        # [-1, 1024, 89] output is [-1, 128, 256].
        c_pre_process = []
        c_pre_process.append(layers.Conv1D(256, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(256, kernel_size=36, strides=2, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        self.c_pre_process = keras.Sequential(c_pre_process)

        # Concatenate processed conditioning and spectrogram in the last
        # channel, giving an input of shape [-1, 128, 256, 2].
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

    def call(self, x_in, c_in):
        """Produces a critic score for x_in given the conditioning
        information c_in.

        Args:
            x_in: A batch of real or generated data. Expected shape is
                [batch_size, time, frequency]
            c_in: The asociated conditioning information. Expected shape
                is [batch_size, 1024, 89].

        Returns:
            The critic scores. Shape is [batch_size, 1]
        """

        x_in = tf.reshape(x_in, self.in_shape)

        c_pre_processed = self.c_pre_process(c_in)
        c_pre_processed = tf.expand_dims(c_pre_processed, 3)
        xc_in = tf.concat([x_in, c_pre_processed], axis=-1)
        return self.l(xc_in)
