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
import soundfile as sf
import matplotlib.pyplot as plt
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.structures import learned_basis_function

MAESTRO_PATH = 'data/MAESTRO_6h.npz'
SAMPLE_RATE = 16000
DECOMPOSITION_IDX = 100

MODELS = {
    'Perceptual': {
        'num_basis_functions': 512,
        'basis_function_legnth': 32,
        'checkpoint_path':\
            '_results/learned_decomposition/percep_loss/training_checkpoints/ckpt-8',
    },
    'Classifier': {
        'num_basis_functions': 512,
        'basis_function_legnth': 32,
        'checkpoint_path':\
            '_results/learned_decomposition/classifier/training_checkpoints/ckpt-10',
    },
    'Critic': {
        'num_basis_functions': 512,
        'basis_function_legnth': 32,
        'checkpoint_path':\
            '_results/learned_decomposition/L2_and_critic/training_checkpoints/ckpt-10',
    }
}

def sort_basis_functions(basis_functions):
    """Sorts a set of basis functions by their distance to the
    function with the smallest two-norm.

    Args:
        basis_functions: The set of basis functions to sort.
            Expected shape is (-1, basis_function_length).

    Returns:
        sorted_basis: The sorted basis functions
        sorted_ids: Mapping from unsorted basis function ids to
            their sorted position.
    """

    min_norm_idx = np.argmin(np.linalg.norm(basis_functions, axis=-1), axis=0)
    min_norm_fn = basis_functions[min_norm_idx]

    ids = list(range(len(basis_functions)))
    sorted_ids = sorted(ids, key=lambda x: np.linalg.norm(basis_functions[x] - min_norm_fn))
    sorted_basis = np.array(basis_functions)[sorted_ids]

    return sorted_basis, sorted_ids

def save_plot(functions, fn_name):
    """Plots a give set of learned basis functions or decomposition.

    Args:
        functions: The set of basis functions to plot or basis function
            decomposition.
        fn_name: The file name for the figure.
    """

    plt.imshow(np.transpose(functions), interpolation='nearest', origin='lower', cmap='bwr')
    plt.colorbar(orientation='horizontal')
    plt.savefig(fn_name, bbox_inches='tight')
    plt.clf()

def plot_basis_functions_frequency_domain(basis_functions, file_name):
    """Plots a give set of learned basis functions in the frequency domain.

    Args:
        basis_functions: The set of basis functions to plot.
        fn_name: The file name for the figure.
    """

    frequency_basis_functions = np.abs(np.fft.rfft(basis_functions))
    save_plot(frequency_basis_functions, file_name)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = waveform_dataset.get_waveform_dataset(MAESTRO_PATH)

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

        spectogram_like = encoder(raw_maestro[DECOMPOSITION_IDX:DECOMPOSITION_IDX+1])
        reconstructed = decoder(spectogram_like)

        sf.write('{}_orig.wav'.format(model), raw_maestro[DECOMPOSITION_IDX], SAMPLE_RATE)
        sf.write('{}_reconstructed.wav'.format(model), tf.squeeze(reconstructed), SAMPLE_RATE)

        encoder_basis = tf.transpose(tf.squeeze(encoder.conv_layer.kernel))
        decoder_basis = tf.transpose(tf.squeeze(decoder.transpose_conv_layer.conv_2d.kernel))

        sorted_encoder_basis, sorted_encoder_ids = sort_basis_functions(encoder_basis)
        sorted_decoder_basis, _ = sort_basis_functions(decoder_basis)

        save_plot(sorted_encoder_basis, '{}_encoder.png'.format(model))
        save_plot(sorted_decoder_basis, '{}_decoder.png'.format(model))

        plot_basis_functions_frequency_domain(
            sorted_encoder_basis, '{}_encoder_frequency.png'.format(model)
        )
        plot_basis_functions_frequency_domain(
            sorted_decoder_basis, '{}_decoder_frequency.png'.format(model)
        )

        spectogram_like = tf.squeeze(spectogram_like)
        spectogram_like = np.array(spectogram_like)[:, sorted_encoder_ids]
        save_plot(spectogram_like, '{}_decomposition.png'.format(model))

if __name__ == '__main__':
    main()
