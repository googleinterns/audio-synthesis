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

"""Training Script for STFTGAN on a waveform dataset.

Follows the same setup as SpecPhaseGAN, but
generates STFTs instead of Magnitude and Instantaneous
Frequency.
"""

import os
import tensorflow as tf
import numpy as np
from audio_synthesis.structures import spec_gan
from audio_synthesis.models import wgan
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.utils import waveform_save_helper as save_helper
from audio_synthesis.utils import spectral
from audio_synthesis.utils import improved_representation

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 16
EPOCHS = 300
SAMPLING_RATE = 16000
N_FILTERS = 2048
FILTER_LENGTH = 256
FILTER_STEP = 128
Z_IN_SHAPE = [4, 64, 1024]
SPECTOGRAM_IMAGE_SHAPE = [-1, 128, 2048, 1]
CHECKPOINT_DIR = '/vol/projects/braithdani/representation_study/MAESTRO/PIRGAN_HR/training_checkpoints/'
RESULT_DIR = '/vol/projects/braithdani/representation_study/MAESTRO/PIRGAN_HR/audio/'
DATASET_PATH = '/vol/projects/braithdani/MAESTRO/MAESTRO_6h.npz'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    filterbank = improved_representation.get_filterbank(N_FILTERS, FILTER_LENGTH, SAMPLING_RATE / 2.0)
    filterbank = filterbank.T
    inv_filterbank = np.linalg.pinv(filterbank)
    process_fn = lambda x: improved_representation.apply_filterbank(
        x, FILTER_LENGTH, FILTER_STEP, filterbank
    )
    raw_dataset = waveform_dataset.get_fb_dataset(
        DATASET_PATH, process_fn
    )
    print(raw_dataset.shape)

    generator = spec_gan.Generator(channels=1, in_shape=Z_IN_SHAPE)
    discriminator = spec_gan.Discriminator(input_shape=SPECTOGRAM_IMAGE_SHAPE)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda representation:\
        improved_representation.reconstruct(
            representation, FILTER_LENGTH, FILTER_STEP, inv_filterbank
        )

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE, RESULT_DIR, get_waveform
        )

    pir_gan_model = wgan.WGAN(
        raw_dataset, generator, [discriminator], Z_DIM,
        generator_optimizer, discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
        fn_save_examples=save_examples
    )

    pir_gan_model.train()

if __name__ == '__main__':
    main()
