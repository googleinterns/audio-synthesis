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

"""Training script for the learned basis function decomposition.
The loss is just the L2 error.
"""

import os
import librosa
import numpy as np
import tensorflow as tf
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.models import learned_basis_decomposition
from audio_synthesis.structures import learned_basis_function

FILTER_LENGTH = 512
NUMBER_OF_FILTERS = 256
STRIDE = 128
BATCH_SIZE = 64
EPOCHS = 100
DATASET_PATH = 'data/SpeechMNIST_1850.npz'
CHECKPOINT_DIR = '_results/learned_decomposition/GFB_decoder/training_checkpoints/'
RESULTS_DIR = '_results/learned_decomposition/GFB_decoder/audio/'

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = waveform_dataset.get_waveform_dataset(DATASET_PATH)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    encoder = learned_basis_function.MPGFBEncoder(FILTER_LENGTH, NUMBER_OF_FILTERS, STRIDE)
    decoder = learned_basis_function.Decoder(FILTER_LENGTH, NUMBER_OF_FILTERS, STRIDE)

    learned_decomposition_model = learned_basis_decomposition.LearnedBasisDecomposition(
        encoder, decoder, optimizer, raw_maestro, BATCH_SIZE, EPOCHS, CHECKPOINT_DIR, RESULTS_DIR
    )
    
    learned_decomposition_model.train()


if __name__ == '__main__':
    main()