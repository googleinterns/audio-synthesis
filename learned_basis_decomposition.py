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
import tensorflow as tf
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models import learned_basis_decomposition
from audio_synthesis.structures import learned_basis_function

FILTER_LENGTH = 32
NUMBER_OF_FILTERS = 512
BATCH_SIZE = 64
EPOCHS = 100
MAESTRO_PATH = 'data/MAESTRO_6h.npz'
CHECKPOINT_DIR = '_results/learned_decomposition/L1_err_w_noise/training_checkpoints/'
RESULTS_DIR = '_results/learned_decomposition/L1_err_w_noise/audio/'

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    encoder = learned_basis_function.Encoder(FILTER_LENGTH, NUMBER_OF_FILTERS)
    decoder = learned_basis_function.Decoder(FILTER_LENGTH)

    learned_decomposition_model = learned_basis_decomposition.LearnedBasisDecomposition(
        encoder, decoder, optimizer, raw_maestro, BATCH_SIZE, EPOCHS, CHECKPOINT_DIR, RESULTS_DIR
    )

    learned_decomposition_model.train()

if __name__ == '__main__':
    main()
