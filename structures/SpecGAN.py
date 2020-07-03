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

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, Reshape, AveragePooling1D, Flatten
from tensorflow.keras.activations import linear
from tensorflow.keras import Model, Sequential

class Generator(Model):
    def __init__(self, channels=1, d=4, activation=linear):
        super(Generator, self).__init__()
        
        self.activation = activation
        
        layers = []
        layers.append(Dense(4 * 8 * 1024))
        layers.append(Reshape((4, 8, 1024)))
        layers.append(ReLU())
        layers.append(Conv2DTranspose(filters=512, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(ReLU())
        layers.append(Conv2DTranspose(filters=256, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(ReLU())
        layers.append(Conv2DTranspose(filters=128, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(ReLU())
        layers.append(Conv2DTranspose(filters=64, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(ReLU())
        layers.append(Conv2DTranspose(filters=channels, kernel_size=(6,6), strides=(2,2), padding='same'))
        #layers.append(ReLU())
        #layers.append(Conv2DTranspose(filters=channels, kernel_size=(6,6), strides=(1,2), padding='same'))
        
        self.l = Sequential(layers)
        
    def call(self, z):
        return self.activation(self.l(z))
        
class Discriminator(Model):
    def __init__(self, d=4):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(Conv2D(filters=64, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv2D(filters=128, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv2D(filters=256, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv2D(filters=512, kernel_size=(6,6), strides=(2,2), padding='same'))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Conv2D(filters=1024, kernel_size=(6,6), strides=(1,2), padding='same'))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Flatten())
        layers.append(Dense(1))
        
        self.l = Sequential(layers)
        
    def call(self, x):
        return self.l(x)
    