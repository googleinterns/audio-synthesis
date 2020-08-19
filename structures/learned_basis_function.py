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

"""Structures for the learned basis decomposition experiments.
Based on the TasNet learned decomposition setup.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from audio_synthesis.utils import layers as layer_util
from audio_synthesis.third_party import gammatone_filterbank

# Controlls the amount of overlap between adjacent windows.
# An overlap factor of 2 means a 50 percent overlap, this is following
# the Conv TasNet paper.
_OVERLAP_FACTOR = 2

class Encoder(keras.Model):
    """The encoder model for the learned decomposition,
    overlap between decomposed winodws is 50% by default
    following the Conv-TasNet paper.
    """

    def __init__(self, length, num_filters):
        """The initiliazer for the Encoder model.

        Args:
            length: The length (in samples) of the filters.
            num_filters: The number of filters.
        """

        super(Encoder, self).__init__()

        self.length = length
        self.stride = self.length // _OVERLAP_FACTOR
        self.num_filters = num_filters

        self.conv_layer = layers.Conv1D(
            filters=num_filters, kernel_size=length, strides=self.stride,
            use_bias=False, padding='SAME'
        )

    def call(self, x_in):
        """Applies the Encoder model, decomposing the input signals
        onto the learned basis functions.
        
        Args:
            x_in: A batch of time domain signals. Expected shape is
                (batch_size, signal_duration, 1).
                
        Returns:
            The signals decomposed onto the encoder basis functions.
            Shape is (batch_size, signal_duration // self.stride, num_filters),
            where signal_duration // self.stride is integer division, rounded up.
        """

        x_in = tf.expand_dims(x_in, axis=2)

        encoded_signals = self.conv_layer(x_in)
        return tf.nn.relu(encoded_signals)
    
class MPGFBEncoder(keras.Model):
    def __init__(self, length, num_filters, stride=128, sr=16000.0):
        super(MPGFBEncoder, self).__init__()
        
        self.length = length
        self.num_filters = num_filters
        self.stride = stride
        self.sr = sr
        self.filterbank = gammatone_filterbank.generate_mpgtf(sr, length/sr, num_filters)
        self.filterbank = np.transpose(self.filterbank)
        self.filterbank = np.expand_dims(self.filterbank, 1).astype(np.float32)
        
    def call(self, x_in):
        x_in = tf.expand_dims(x_in, axis=2)
        encoded_signals = tf.nn.conv1d(x_in, tf.stop_gradient(self.filterbank), stride=self.stride, padding='SAME')
        return tf.nn.relu(encoded_signals)

class Decoder(keras.Model):
    """The decoder model for the learned basis decomposition."""

    def __init__(self, length, num_filters, stride, sr=16000.0):
        """The initilizer for the Decoder model.

        Args:
            length: The length of the filters.
        """

        super(Decoder, self).__init__()

        self.length = length
        self.stride = stride
        
        filterbank = gammatone_filterbank.generate_mpgtf(sr, length/sr, num_filters).astype(np.float32)
        filterbank_inv = tf.linalg.pinv(tf.expand_dims(filterbank, 0))[0]
        conv_filterbank_inv = tf.expand_dims(filterbank_inv, 1)
        
        self.filterbank = tf.Variable(initial_value=conv_filterbank_inv, trainable=True)
        
    
        #self.transpose_conv_layer = layer_util.Conv1DTranspose(
        #    1, kernel_size=length, strides=self.stride,
        #    use_bias=False, padding='SAME'
        #)

    def call(self, x_in):
        """Applys the decoder function to the input. Reconstructing the
        signal domain representation.
        
        Args:
            x_in: Signals in a decomposed representation. 
                Shape is (batch_size, signal_duration // self.stride,
                num_filters).
            
        Returns:
            A batch of time-domain waveforms. Shape is
            (batch_size, signal_duration, 1).
        """
        
        reconstructed = tf.nn.conv1d_transpose(
            x_in, self.filterbank, (x_in.shape[0], self.stride * x_in.shape[1], 1),
            strides=self.stride, padding='SAME'
        )
        return reconstructed
        #return self.transpose_conv_layer(x_in)

class NLDecoder(keras.Model):
    """The decoder model for the learned basis decomposition."""

    def __init__(self, length, num_filters, stride=128):
        """The initilizer for the Decoder model.

        Args:
            length: The length of the filters.
        """

        super(NLDecoder, self).__init__()

        self.length = length
        self.stride = stride
        self.num_filters = num_filters

        sequential = []
        sequential.append(layer_util.Conv1DTranspose(filters=self.num_filters, strides=self.stride, kernel_size=length))
        sequential.append(layers.LeakyReLU())
        #sequential.append(layer_util.Conv1DTranspose(filters=self.num_filters//2, strides=1, kernel_size=length))
        #sequential.append(layers.LeakyReLU())
        #sequential.append(layer_util.Conv1DTranspose(filters=self.num_filters//4, strides=1, kernel_size=length))
        #sequential.append(layers.ReLU())
        #sequential.append(layer_util.Conv1DTranspose(filters=self.num_filters//4, strides=1, kernel_size=length))
        #sequential.append(layers.LeakyReLU())
        sequential.append(layer_util.Conv1DTranspose(filters=1, strides=1, kernel_size=length))
        
        self.l = keras.Sequential(sequential)

    def call(self, x_in):
        """Applys the decoder function to the input. Reconstructing the
        signal domain representation.
        
        Args:
            x_in: Signals in a decomposed representation. 
                Shape is (batch_size, signal_duration // self.stride,
                num_filters).
            
        Returns:
            A batch of time-domain waveforms. Shape is
            (batch_size, signal_duration, 1).
        """
        
        return self.l(x_in)
    
class Classifier(keras.Model):
    """The classifier used in some learned decomposition experiments.
    Used for classifing the midi data from the decomposed representation.
    Multiple keys can be active at once. Hence, we use a sigmoid activation,
    not softmax.
    """

    def __init__(self, num_keys, structure=[512, 128, 100]):
        """The initilizer for the Classifier model.

        Args:
            num_keys: The number of musical notes to classify in the
                midi data.
            structure: An array of dense layer sizes (int), one for
                each dense layer in the classifier structure.
        """

        super(Classifier, self).__init__()
        self.num_keys = num_keys
        self.structure = structure

        classifier = []
        for layer_size in self.structure:
            classifier.append(layers.Dense(layer_size))
            classifier.append(layers.ReLU())
        classifier.append(layers.Dense(num_keys))

        self.classifier = keras.Sequential(classifier)

    def call(self, blocks):
        """Takes in decomposed signal representation and produces a
        classification.
        
        Args:
            blocks: The blocks to be classified. Shape is
                (-1, self.num_keys)
                
        Returns:
            logits: The logit output from the classifier.
            probabilities: The probability output from the 
                classifier, logits after a sigmoid activation.
        """

        logits = self.classifier(blocks)
        probabilities = tf.nn.sigmoid(logits)

        return logits, probabilities

class Discriminator(keras.Model):
    """The Discriminator model used in some learned basis function
    experiments. Used as an auxiliary loss component.
    """

    def __init__(self, name='discriminator'):
        super(Discriminator, self).__init__()
        sequential = []
        sequential.append(layers.Conv1D(32, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(64, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(128, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Conv1D(256, kernel_size=36, strides=4, padding='same'))
        sequential.append(layers.LeakyReLU(alpha=0.2))
        sequential.append(layers.Flatten())
        sequential.append(layers.Dense(1))

        self.discriminator = keras.Sequential(sequential, name=name)

    def call(self, x_in):
        """Applies the discriminator network.
        
        Args:
            x_in: A batch of time-domain waveforms. Expected shape
                is (batch_size, signal_duration, 1).
                
        Returns:
            Real-valued scores, according to the WGAN implementation.
            Shape is (batch_size, 1).
        """

        output = self.discriminator(x_in)
        return output
