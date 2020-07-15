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
for GAN-TTS [https://arxiv.org/abs/1909.11646]. This implementation is
based off the paper. This implementation only contains the unconditional random window
discriminators.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from audio_synthesis.utils import layers as layers_util


class Generator(keras.Model): # pylint: disable=too-many-ancestors
    """The GAN-TTS Generator Function.

    Composes a number of GBlocks according to the structure
    reported in the GAN-TTS paper.
    """

    def __init__(self, g_block_configs, latent_shape):
        """Initlizer for the GAN-TTS Generator.

        Paramaters:
            g_block_configs: Contains N sub arrays for
                each generator block. Each sub array is of
                the form [input channels, output channels,
                upsample factor]
                    - input channels: The number of input channels
                        to the block
                    - output channels: The number of channels in the
                        block's output.
                    - upsample factor: The amout to upsample in the
                        time domain.
            latent_shape: The shape of the latent features given to the
                first GBlock. The latent features will be upscaled and
                reshaped.
        """
        super(Generator, self).__init__()

        self.pre_process = keras.Sequential([
            layers.Dense(np.prod(latent_shape)),
            layers.Reshape(latent_shape),
            layers.Conv1D(strides=1, kernel_size=3, filters=g_block_configs[0][0], padding='same')
        ])

        g_blocks = []
        for config in g_block_configs:
            input_channels, output_channels, upsample_factor = config
            g_block = GBlock(input_channels=input_channels,
                             output_channels=output_channels,
                             upsample_factor=upsample_factor)
            g_blocks.append(g_block)
        self.g_blocks = keras.Sequential(g_blocks)

        self.post_process = layers.Conv1D(strides=1, kernel_size=3, filters=1, padding='same')

    def call(self, z_in, training=False):
        output = self.pre_process(z_in)
        output = self.g_blocks(output)
        output = self.post_process(output)

        return output

class GBlock(keras.Model): # pylint: disable=too-many-ancestors
    """Implementation of the GBlock component that makes up the GAN-TTS generator

    This implementaion, for the most part, follows Figure 1 of the paper
    [https://arxiv.org/pdf/1909.11646.pdf]. However, the conditioning information
    is excluded.
    """

    def __init__(self, input_channels, output_channels, upsample_factor):
        """Initilizer for the GBlock component of the GAN-TTS generator.

        Paramaters:
            input_channels: The number of channels given as input.
            output_channels: The number of output channels.
            upsampling_factor: The factor to upscale the time dimention by.
        """
        super(GBlock, self).__init__()

        self.stack_1 = keras.Sequential([
            layers.ReLU(),
            layers_util.Conv1DTranspose(
                filters=input_channels, kernel_size=upsample_factor * 2, strides=upsample_factor,
                padding='same'
            ),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, padding='same'
            )
        ])

        self.stack_2 = keras.Sequential([
            layers.ReLU(),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=2,
                padding='same'
            )
        ])

        self.residual_1 = keras.Sequential([
            layers_util.Conv1DTranspose(
                filters=input_channels, kernel_size=upsample_factor * 2, strides=upsample_factor,
                padding='same'
            ),
            layers.Conv1D(filters=output_channels, kernel_size=1, strides=1, padding='same')
        ])

        self.stack_3 = keras.Sequential([
            layers.ReLU(),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=4,
                padding='same'
            )
        ])

        self.stack_4 = keras.Sequential([
            layers.ReLU(),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=8,
                padding='same'
            )
        ])

    def call(self, x):
        stack_1_out = self.stack_1(x)
        stack_2_out = self.stack_2(stack_1_out)

        residual = self.residual_1(x)
        residual_output = stack_2_out + residual

        stack_3_out = self.stack_3(residual_output)
        stack_4_out = self.stack_4(stack_3_out)

        return stack_4_out + residual_output

class Discriminator(keras.Model): # pylint: disable=too-many-ancestors
    """Implementation of the (unconditional) Random
    Window Discriminators for GAN-TTS.
    """

    def __init__(self, window_sizes=(240, 480, 960, 1920, 3600),
                 factors=((5, 3), (5, 3), (5, 3), (5, 3), (2, 2)),
                 omega=240):
        """Initilizer for the unconditional disctiminator
        component of the GAN-TTS model.

        Paramaters:
            window_sizes: An array containing the windows sizes of each
                random window discriminator (in samples).
            factors: The downsample factors for each of the DBlocks in
                each random window dicriminator.
            omega: The factor at which samples are moved into the channels
                to ensure a constant temporal length.
        """
        super(Discriminator, self).__init__()

        self.omega = omega
        self.window_sizes = window_sizes

        discriminators = []
        for factor in factors:
            discriminators.append(UnconditionalDBlocks(factors=factor))

        self.discriminators = discriminators


    def call(self, x_in):
        scores = []
        # For each block size
        for i, window_size in enumerate(self.window_sizes):
            # Select random sub-window
            idx = tf.random.uniform((1,), 0,
                                    (x_in.shape[1] - window_size),
                                    tf.dtypes.int32)[0]
            x_windowed = x_in[:, idx : idx + window_size, :]

            # Move samples into channels to ensure constant temporal length
            downsampling_factor = window_size // self.omega
            assert window_size // self.omega == int(window_size // self.omega)

            x_reshaped = tf.reshape(x_windowed, (-1, self.omega, downsampling_factor))

            score = self.discriminators[i](x_reshaped)
            scores.append(score)

        return scores

class UnconditionalDBlocks(keras.Model): # pylint: disable=too-many-ancestors
    """Implementation of a Unconditional Random Window Discriminator.

    A collection of sequential DBlocks following Figure 2 of the
    paper [https://arxiv.org/pdf/1909.11646.pdf].
    """

    def __init__(self, factors=(5, 3), output_channels=(128, 256)):
        """Initilizer for an unconditional random window discriminator
        component of the GAN-TTS discriminator.

        Paramaters:
            factors: An array containing a downsampling factor for each DBlock.
            output_channels: An array containing the number of output channels
                for each DBlock.
        """
        super(UnconditionalDBlocks, self).__init__()

        dblocks = []
        dblocks.append(DBlock(64, 1))
        for i, factor in enumerate(factors):
            dblocks.append(DBlock(output_channels[i], factor))
        dblocks.append(DBlock(output_channels[-1], 1))
        dblocks.append(DBlock(output_channels[-1], 1))
        self.dblocks = keras.Sequential(dblocks)


    def call(self, x_in):
        return self.dblocks(x_in)

class DBlock(keras.Model): # pylint: disable=too-many-ancestors
    """Implementation of the DBlock for GAN-TTS Discriminators.

    The implementation of this block follows Figure 1 of
    the paper [https://arxiv.org/pdf/1909.11646.pdf].
    """

    def __init__(self, output_channels, downsample_factor):
        """Initilizer for the unconditional DBlock of the
        GAN-TTS discriminator.

        Paramaters:
            output_channels: The number of output channels
                from the DBlock.
            downsample_factor: The downsampling factor of the
                DBlock.
        """

        super(DBlock, self).__init__()

        self.stack = keras.Sequential([
            layers.AveragePooling1D(pool_size=downsample_factor, strides=downsample_factor),
            layers.ReLU(),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=1,
                padding='same'
            ),
            layers.ReLU(),
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=2,
                padding='same'
            )
        ])

        self.residual = keras.Sequential([
            layers.Conv1D(
                filters=output_channels, kernel_size=3, strides=1, dilation_rate=1,
                padding='same'
            ),
            layers.AveragePooling1D(pool_size=downsample_factor, strides=downsample_factor)
        ])

    def call(self, x_in):
        stack_output = self.stack(x_in)
        residual_output = self.residual(x_in)

        return stack_output + residual_output
