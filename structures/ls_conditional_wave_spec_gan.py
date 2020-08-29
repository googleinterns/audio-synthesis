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
of the Conditional WaveSpecGAN. Designed for last second conditioning, i.e.,
conditioning informtion is the last (half) second of generated audio.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """The Generator function for the last second
    Conditional WaveSpecGAN.
    """

    def __init__(self):
        """Initilizer function for the Condtional WaveSpecGAN
        generator function.
        """

        super(Generator, self).__init__()

        c_pre_process = []
        c_pre_process.append(layers.Conv1D(64, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(256, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(512, kernel_size=36, strides=4, padding='same'))
        c_pre_process.append(layers.LeakyReLU(alpha=0.2))
        c_pre_process.append(layers.Conv1D(512, kernel_size=36, strides=2, padding='same'))
        self.encoder = keras.Sequential(c_pre_process)

        z_pre_process = []
        z_pre_process.append(layers.Dense(16 * 512))
        z_pre_process.append(layers.Reshape((16, 512)))
        self.z_preprocess = keras.Sequential(z_pre_process)

        sequential = []
        sequential.append(layer_utils.Conv1DTranspose(filters=512, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=256, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=128, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=64, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=1, strides=4, kernel_size=36))

        self.l = keras.Sequential(sequential)

    def call(self, z_in, c_in):
        """Generates audio from the given conditioning information and
        random noise.

        Args:
            z_in: The random noise vectors. Expected shape
                is [batch_size, z_dim].
            c_in: The conditioning information. Expected shape
                is [batch_size, c_signal_length].

        Returns:
            The generated audio of shape
            [batch_size, signal_length, 1]
        """

        z_pre_processed = self.z_preprocess(z_in)

        c_in = tf.expand_dims(c_in, 2)
        c_in_enc = self.encoder(c_in)
        zc_in = tf.concat([z_pre_processed, c_in_enc], axis=-1)
        output = self.l(zc_in)
        return output

class WaveformDiscriminator(keras.Model):
    """Implementation of the discriminator for
    Conditional WaveSpecGAN.
    """

    def __init__(self, input_shape, weighting=1.0):
        """Inilizer function for the waveform discriminator component
        of the conditional WaveSpecGAN.

        Args:
            input_shape: The required shape for the data input 
                to the discriminator.
            weighting: Relative weighting of the discriminator in the 
                loss. Default 1.0.
        """

        super(WaveformDiscriminator, self).__init__()

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

        self.l = keras.Sequential(sequential)

    def call(self, x_in, c_in):
        """Produces a critic score for the input given the conditioning
        information.

        Args:
            x_in: A batch of real or generated data. Expected shape is
                [batch_size, signal_length]
            c_in: The batch of asociated conditioning information.
                Expected shape is [batch_size, c_signal_length]

        Returns:
            Real valued scores. Shape is [batch_size, 1].
        """

        x_in = tf.reshape(x_in, self.in_shape)

        xc_in = tf.concat([x_in, c_in], axis=-2)
        output = self.l(xc_in)
        return output

class SpectogramDiscriminator(keras.Model):
    """Implementation of the SpecGAN Discriminator Function."""

    def __init__(self, input_shape, weighting=1.0):
        """Initilizer for the spectogram discriminator component of
        the conditional WaveSpecGAN.

        Args:
            input_shape: The required shape for data at input to
                the discriminator.
            weighting: The relative weighting of the discriminator in
                the final loss.
        """

        super(SpectogramDiscriminator, self).__init__()

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

    def call(self, x_in, c_in):
        """Produces a critic score for x_in given the conditioning
        information c_in.

        Args:
            x_in: A batch of real or generated data. Expected shape is
                [batch_size, time, frequency].
            c_in: The asociated conditioning information. Expected shape
                is [batch_size, c_time, frequency].

        Returns:
            The real valued discriminator scores. Shape is [batch_size, 1].
        """

        x_in = tf.reshape(x_in, self.in_shape)

        xc_in = tf.concat([x_in, c_in], axis=-3)
        return self.l(xc_in)
