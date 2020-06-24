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

"""An implementation of the Generator and Discriminator for the GAN-TTS model.

This file contains an implementation of the Generator and Discriminator networks
for GAN-TTS [https://arxiv.org/abs/1909.11646]. This implementation is soley
based off the paper. This implementation only contains the unconditional random window
discriminators.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Conv1D, Conv2DTranspose, Reshape, AveragePooling1D
from tensorflow.keras import Model, Sequential
from utils.Layers import Conv1DTranspose
import numpy as np


class Generator(Model):
    """
    """
    
    def __init__(self, g_block_configs, latent_shape):
        super(Generator, self).__init__()
        
        self.pre_process = Sequential([
            Dense(np.prod(latent_shape)),
            Reshape(latent_shape),
            Conv1D(strides=1, kernel_size=3, filters=g_block_configs[0][0], padding='same')
        ])

        g_blocks = []
        for config in g_block_configs:
            input_channels, output_channels, upsample_factor = config
            g_block = GBlock(input_channels=input_channels, output_channels=output_channels, upsample_factor=upsample_factor)
            g_blocks.append(g_block)
        self.g_blocks = Sequential(g_blocks)

        self.post_process = Conv1D(strides=1, kernel_size=3, filters=1, padding='same')

    def call(self, z):
        x = self.pre_process(z)
        x = self.g_block(x)
        x = self.post_process(x)

        return x


class GBlock(Model):
    """Implementation of the GBlock component that makes up the GAN-TTS generator
    
    This implementaion, for the most part, follows Figure 1 of the paper
    [https://arxiv.org/pdf/1909.11646.pdf]. However, the conditioning information
    is excluded.
    """
    
    def __init__(self, input_channels, output_channels, upsample_factor):
        super(GBlock, self).__init__()
        
        self.stack_1 = Sequential([
            ReLU(),
            Conv1DTranspose(filters=input_channels, kernel_size=upsample_factor*2, strides=upsample_factor, padding='same'),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, padding='same')                       
        ])

        self.stack_2 = Sequential([
            ReLU(),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=2, padding='same')    
        ])

        self.residual_1 = Sequential([
            Conv1DTranspose(filters=input_channels, kernel_size=upsample_factor*2, strides=upsample_factor, padding='same'),
            Conv1D(filters=output_channels, kernel_size=1, strides=1, padding='same') 
        ])

        self.stack_3 = Sequential([
            ReLU(),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=4, padding='same')
        ])

        self.stack_4 = Sequential([
            ReLU(),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=8, padding='same')
        ])

    def call(self, x):
        stack_1_out = self.stack_1(x)
        stack_2_out = self.stack_2(stack_1_out)

        residual = self.residual_1(x)
        residual_output = stack_2_out + residual

        stack_3_out = self.stack_3(residual_output)
        stack_4_out = self.stack_4(stack_3_out)

        return stack_4_out + residual_output



class Discriminator(Model):
    """Implementation of the (unconditional) Random Window Discriminators for GAN-TTS.
    """
    
    def __init__(self, window_sizes=(240, 480, 960, 1920, 3600), factors=((5,3),(5,3),(5,3),(5,3),(2,2)), omega=240):
        super(Discriminator, self).__init__()
        
        self.omega = omega
        self.window_sizes = window_sizes

        discriminators = []
        for ws, f in zip(window_sizes, factors):
            discriminators.append(UnconditionalDBlocks(factors=f))
        
        self.discriminators = discriminators


    def call(self, x):
        scores = []
        # For each block size
        for i, ws in enumerate(self.window_sizes):
            # Select random sub-window
            idx = tf.random.uniform(shape=(1,), minval=0, maxval=(x.shape[1] - ws), dtype=tf.dtypes.int32)[0]
            x_windowed = x[:,idx:idx+ws,:]
            
            # Move samples into channels to ensure constant temporal length
            downsampling_factor = ws // self.omega
            x_reshaped = tf.reshape(x_windowed, (-1, self.omega, downsampling_factor))
        
            score = self.discriminators[i](x_reshaped)
            scores.append(score)
        
        return scores

class UnconditionalDBlocks(Model):
    """Implementation of a Unconditional Random Window Discriminator.
    
    A collection of sequential DBlocks following Figure 2 of the
    paper [https://arxiv.org/pdf/1909.11646.pdf].
    """
    
    def __init__(self, factors=(5,3), output_channels=(128, 256)):
        super(UnconditionalDBlocks, self).__init__()
        
        dblocks = []
        dblocks.append(DBlock(64, 1))
        for i in range(len(factors)):
            dblocks.append(DBlock(output_channels[i], factors[i]))
        dblocks.append(DBlock(output_channels[-1], 1))
        dblocks.append(DBlock(output_channels[-1], 1))
        self.dblocks = Sequential(dblocks) 


    def call(self, x):
        return self.dblocks(x)



class DBlock(Model):
    """Implementation of the DBlock component that makes up the unconditional GAN-TTS Discriminators.
    
    The implementation of this block follows Figure 1 of
    the paper [https://arxiv.org/pdf/1909.11646.pdf].
    """
    
    def __init__(self, output_channels, downsample_factor):
        super(DBlock, self).__init__()
        
        self.stack = Sequential([
            AveragePooling1D(pool_size=downsample_factor, strides=downsample_factor), # 100% sure that this is how we should be downsampling
            ReLU(),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=1, padding='same'),
            ReLU(),
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=2, padding='same')  
        ])

        self.residual = Sequential([
            Conv1D(filters=output_channels, kernel_size=3, strides=1, dilation_rate=1, padding='same'),
            AveragePooling1D(pool_size=downsample_factor, strides=downsample_factor)
        ])

    def call(self, x):
        so = self.stack(x) 
        ro = self.residual(x)
        
        return so + ro
