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

"""
"""

import tensorflow as tf
from tensorflow.keras import activations, layers
from tensorflow import keras
import audio_synthesis.utils.layers as layer_utils

class Generator(keras.Model):
    """The Generator function for Conditional WaveSpecGAN.
    """

    def __init__(self):
        super(Generator, self).__init__()
        sequential = [] 
        sequential.append(layers.Conv1D(filters=1024, strides=1, kernel_size=36, padding='same'))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=512, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=256, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=128, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=64, strides=2, kernel_size=36))
        sequential.append(layers.ReLU())
        sequential.append(layer_utils.Conv1DTranspose(filters=1, strides=2, kernel_size=36))

        self.l = keras.Sequential(sequential)

    def call(self, c_in):
        output = self.l(c_in)
        return output


class WaveformDiscriminator(keras.Model):
    """Implementation of the discriminator for Conditional WaveSpecGAN
    """

    def __init__(self):
        super(WaveformDiscriminator, self).__init__()
        
        conditional_sequental = []
        conditional_sequental.append(layers.Conv1D(128, kernel_size=36, strides=2, padding='same'))
        conditional_sequental.append(layers.LeakyReLU(alpha=0.2))
        conditional_sequental.append(layers.Conv1D(256, kernel_size=36, strides=2, padding='same'))
        conditional_sequental.append(layers.LeakyReLU(alpha=0.2))
        self.sequential_conditional = keras.Sequential(conditional_sequental)
        
        sequential = []
        sequential.append(layers.Conv1D(64, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(256, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(256, kernel_size=36, strides=2, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        self.sequential_waveform = keras.Sequential(sequential)
        
        sequential_joint = []
        sequential_joint.append(layers.Conv1D(512, kernel_size=36, strides=2, padding='same'))
        sequential_joint.append(layers.LeakyReLU(alpha=0.2))
        sequential_joint.append(layers.Conv1D(1024, kernel_size=36, strides=2, padding='same'))
        sequential_joint.append(layers.LeakyReLU(alpha=0.2))
        sequential_joint.append(layers.Flatten())
        sequential_joint.append(layers.Dense(1))

        self.sequential_joint = keras.Sequential(sequential_joint)

    def call(self, x_in, c_in): 
        x_processed = self.sequential_waveform(x_in)
        c_processed = self.sequential_conditional(c_in)
        xc_in = tf.concat([x_processed, c_processed], axis=-1)
        output = self.sequential_joint(xc_in)
        return output


class SpectogramDiscriminator(keras.Model):
    """Implementation of the SpecGAN Discriminator Function."""

    def __init__(self):
        super(SpectogramDiscriminator, self).__init__()
        self.c_pre_process = layers.Conv1D(256, kernel_size=36, strides=4, padding='same')

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
        c_in = tf.squeeze(c_in)
        c_pre_processed = self.c_pre_process(c_in)
        c_pre_processed = tf.expand_dims(c_pre_processed, 3)
        xc_in = tf.concat([x_in, c_pre_processed], axis=-1)
        return self.l(xc_in)
