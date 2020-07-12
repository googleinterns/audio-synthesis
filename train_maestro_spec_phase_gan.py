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

"""Training Script for SpecPhaseGAN on MAESTRO.

This follows the origonal SpecGAN training,
where the magnitude and phase (instantaneous frequency)
are normalized to sit between -1 and 1.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, utils
from audio_synthesis.structures import spec_gan
from audio_synthesis.models import wgan
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import maestro_save_helper as save_helper

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 256
FFT_FRAME_STEP = 128
LOG_MAGNITUDE = True
INSTANTANEOUS_FREQUENCY = True
SPECTOGRAM_IMAGE_SHAPE = [-1, 128, 128, 2]
CHECKPOINT_DIR = '_results/representation_study/SpecPhaseGAN/training_checkpoints/'
RESULT_DIR = '_results/representation_study/SpecPhaseGAN/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro, magnitude_stats, phase_stats =\
        maestro_dataset.get_maestro_magnitude_phase_dataset(
            MAESTRO_PATH, FFT_FRAME_LENGTH, FFT_FRAME_STEP, LOG_MAGNITUDE,
            INSTANTANEOUS_FREQUENCY
        )

    normalized_raw_maestro = []
    pb_i = utils.Progbar(len(raw_maestro))
    for spectogram in raw_maestro:
        norm_mag = maestro_dataset.normalize(spectogram[:, :, 0], *magnitude_stats, *phase_stats)
        norm_phase = maestro_dataset.normalize(spectogram[:, :, 1], *magnitude_stats, *phase_stats)

        norm = np.concatenate([np.expand_dims(norm_mag, axis=2),
                               np.expand_dims(norm_phase, axis=2)], axis=-1)
        normalized_raw_maestro.append(norm)
        pb_i.add(1)

    generator = spec_gan.Generator(channels=2, activation=activations.tanh)
    discriminator = spec_gan.Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda spectogram:\
        save_helper.get_waveform_from_normaized_magnitude(
            spectogram, [magnitude_stats, phase_stats], FFT_FRAME_LENGTH,
            FFT_FRAME_STEP, LOG_MAGNITUDE, INSTANTANEOUS_FREQUENCY
        )

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE, RESULT_DIR, get_waveform
        )

    spec_phase_gan_model = wgan.WGAN(
        normalized_raw_maestro, SPECTOGRAM_IMAGE_SHAPE, generator, discriminator, Z_DIM,
        generator_optimizer, discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
        fn_save_examples=save_examples
    )

    spec_phase_gan_model.train()

if __name__ == '__main__':
    main()
