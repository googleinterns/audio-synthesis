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
implementation of WaveGAN. The key difference between this imlementation
and the offical one is that we do not use Phase Shuffle to avoid checkerboarding,
instead we choose a kernel size of 24 (instead of 25),
such that it is divisible by the stride.

""" 

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Conv1D, Conv2DTranspose, Reshape, AveragePooling1D, Flatten
from tensorflow.keras import Model, Sequential
from utils.Layers import Conv1DTranspose

class Generator(Model):
    """The Generator function for WaveGAN.
    
    The model takes a latent vector as input and transforms it into
    a signal with 16 time-steps and 1024 channels. Six transpose
    convolution layers upscale in the time dimention to 65536 and
    reduce the channel dimention to one. The output is approximatly
    four seconds of 16kHz audio.
    """
    
    def __init__(self, name='generator'):
        super(Generator, self).__init__()
        layers = []
        layers.append(Dense(16 * 1024))
        layers.append(Reshape((16, 1024)))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=512, strides=4, kernel_size=24))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=256, strides=4, kernel_size=24))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=128, strides=4, kernel_size=24))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=64, strides=4, kernel_size=24))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=64, strides=4, kernel_size=24))
        layers.append(ReLU())
        layers.append(Conv1DTranspose(filters=1, strides=4, kernel_size=24))

        self.l = Sequential(layers, name=name)


    def call(self, z):
        x = self.l(z)
        return x

    
class Discriminator(Model):
    """Implementation of the discriminator for WaveGAN
    
    The model takes as input a real or fake waveform and,
    following the WGAN framework, produces a real valued
    output.
    """
    
    def __init__(self, name='discriminator'):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(Conv1D(64, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv1D(128, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv1D(256, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv1D(512, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv1D(1024, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv1D(2014, kernel_size=24, strides=4))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Flatten())
        layers.append(Dense(1))

        self.l = Sequential(layers, name=name)

    def call(self, x):
        output = self.l(x)
        return output