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

"""This module handles generating images of the learned
encoder and decoder basis functions for various experiments.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models import learned_basis_decomposition
from audio_synthesis.structures import learned_basis_function

RANDOM_SIGNAL = np.random.normal(size=(1, 2**14)).astype(np.float32)

MODELS = {
    'Classifier': {
        'num_basis_functions': 1024,
        'basis_function_legnth': 64,
        'checkpoint_path': '_results/learned_decomposition/classifier/training_checkpoints/ckpt-10',
    },
    'Critic': {
        'num_basis_functions': 512,
        'basis_function_legnth': 32,
        'checkpoint_path': '_results/learned_decomposition/L2_and_critic/training_checkpoints/ckpt-10',
    }
}

def plot_basis_functions(basis_functions, fn_name):
    """Plots a give set of learned basis functions.
    
    Args:
        basis_functions: The set of basis functions to plot.
        fn_name: The file name for the figure.
    """

    basis_functions = np.transpose(basis_functions)
    peak_bins = np.argmax(basis_functions, axis=-1)

    ids = list(range(512))
    sorted_ids = sorted(ids, key=lambda x: peak_bins[x])
    sorted_basis = np.array(basis_functions)[sorted_ids]

    plt.imshow(np.transpose(sorted_basis), interpolation='nearest', origin='lower')
    plt.colorbar(orientation='horizontal')
    plt.savefig(fn_name, bbox_inches='tight')
    plt.clf()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    for model in MODELS:
        n_basis_functions = MODELS[model]['num_basis_functions']
        basis_function_length = MODELS[model]['basis_function_legnth']

        encoder = learned_basis_function.Encoder(basis_function_length, n_basis_functions)
        decoder = learned_basis_function.Decoder(basis_function_length)
        
        checkpoint = tf.train.Checkpoint(
            encoder=encoder,
            decoder=decoder
        )
        checkpoint.restore(MODELS[model]['checkpoint_path']).expect_partial()

        decoder(encoder(RANDOM_SIGNAL))

        encoder_basis = tf.squeeze(encoder.enc.kernel)
        decoder_basis = tf.squeeze(decoder.dec.conv_2d.kernel)

        plot_basis_functions(encoder_basis, '{}_encoder.png'.format(model))
        plot_basis_functions(decoder_basis, '{}_decoder.png'.format(model))

if __name__ == '__main__':
    main()
