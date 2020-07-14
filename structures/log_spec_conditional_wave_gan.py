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

"""An implemementation of the Generator and Discriminator for a Conditional WaveGAN.

This file contains an implementation of the generator and discriminator
components for the WaveGAN [https://arxiv.org/abs/1802.04208] model.

The key difference between this implementation
and the offical one is that we do not use Phase Shuffle to avoid checkerboarding,
instead we choose a kernel size of 36 (instead of 25),
such that it is divisible by the stride. In addition, this implementation is designed to
be conditioned on a log magnitude spectrum.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """The Generator function for ConditionalWaveGAN.

    The model takes a log magnitude spectrum and transforms it into a
    waveform.
    """

    def __init__(self, name='generator'):
        super(Generator, self).__init__()
        sequential = []
        #sequential.append(layers.Dense(16 * 1024))
        #sequential.append(layers.Reshape((16, 1024)))
        #sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=512, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=256, strides=4, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=128, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=64, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=1, strides=2, kernel_size=36))

        self.l = keras.Sequential(sequential, name=name)

    def call(self, c_in):
        output = self.l(c_in)
        return output

    
class ConditionalDiscriminator(keras.Model):
    """Implementation of the SpecGAN Discriminator Function."""

    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()

        # Pre Process the Waveform [2**14, 1] -> [128, 256]
        sequential_waveform = []
        sequential_waveform.append(layers.Conv1D(64, kernel_size=36, strides=4, padding='same'))
        sequential_waveform.append(layers.LeakyReLU(alpha=0.2))
        sequential_waveform.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        sequential_waveform.append(layers.LeakyReLU(alpha=0.2))
        sequential_waveform.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        sequential_waveform.append(layers.LeakyReLU(alpha=0.2))
        sequential_waveform.append(layers.Conv1D(128, kernel_size=36, strides=2, padding='same'))
        sequential_waveform.append(layers.LeakyReLU(alpha=0.2))
        self.process_waveform = keras.Sequential(sequential_waveform)
        
        # Pre process the spectrum [128, 128, 1] -> [128, 256, 1]
        sequential_spectrum = []
        sequential_spectrum.append(layers.Conv2D(filters=128, kernel_size=(6, 6),
                                        strides=(1, 1), padding='same'))
        sequential_spectrum.append(layers.LeakyReLU(alpha=0.2))
        sequential_spectrum.append(layers.Conv2D(filters=256, kernel_size=(6, 6),
                                        strides=(1, 1), padding='same'))
        sequential_spectrum.append(layers.LeakyReLU(alpha=0.2))
        sequential_spectrum.append(layers.Conv2D(filters=1, kernel_size=(6, 6),
                                        strides=(1, 1), padding='same'))
        sequential_spectrum.append(layers.LeakyReLU(alpha=0.2))
        self.process_spectrum = keras.Sequential(sequential_spectrum)
        
        # Combine and reduce
        sequential_combined = []
        sequential_combined.append(layers.Conv1D(512, kernel_size=36, strides=2, padding='same'))
        sequential_combined.append(layers.LeakyReLU(alpha=0.2))
        sequential_combined.append(layers.Conv1D(1024, kernel_size=36, strides=2, padding='same'))
        sequential_combined.append(layers.Flatten())
        sequential_combined.append(layers.Dense(1))

        self.process_combined = keras.Sequential(sequential_combined)

    def call(self, x_in, c_in):
        x_processed = self.process_waveform(x_in)
        c_processed = self.process_spectrum(c_in)
        xc_in = tf.concat([x_processed, tf.squeeze(c_processed)], axis=-1)
        return self.process_combined(xc_in)
