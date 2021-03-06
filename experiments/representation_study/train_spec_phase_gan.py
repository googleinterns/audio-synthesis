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

"""Training Script for SpecPhaseGAN on a waveform dataset.

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
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.utils import waveform_save_helper as save_helper

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 2500
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
LOG_MAGNITUDE = True
INSTANTANEOUS_FREQUENCY = False
Z_IN_SHAPE = [4, 8, 1024]
SPECTOGRAM_IMAGE_SHAPE = [-1, 128, 256, 2]
CHECKPOINT_DIR = '_results/representation_study/SpeechMNIST/MagPhaseGAN_HR/training_checkpoints/'
RESULT_DIR = '_results/representation_study/SpeechMNIST/MagPhaseGAN_HR/audio/'
DATASET_PATH = 'data/SpeechMNIST_1850.npz'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    raw_dataset, magnitude_stats, phase_stats =\
        waveform_dataset.get_magnitude_phase_dataset(
            DATASET_PATH, FFT_FRAME_LENGTH, FFT_FRAME_STEP, LOG_MAGNITUDE,
            INSTANTANEOUS_FREQUENCY
        )

    normalized_raw_dataset = []
    pb_i = utils.Progbar(len(raw_dataset))
    for spectogram in raw_dataset:
        norm_mag = waveform_dataset.normalize(spectogram[:, :, 0], *magnitude_stats)
        norm_phase = waveform_dataset.normalize(spectogram[:, :, 1], *phase_stats)

        norm = np.concatenate([np.expand_dims(norm_mag, axis=2),
                               np.expand_dims(norm_phase, axis=2)], axis=-1)
        normalized_raw_dataset.append(norm)
        pb_i.add(1)
    normalized_raw_dataset = np.array(normalized_raw_dataset)

    generator = spec_gan.Generator(channels=2, activation=activations.tanh, in_shape=Z_IN_SHAPE)
    discriminator = spec_gan.Discriminator(input_shape=SPECTOGRAM_IMAGE_SHAPE)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda spectogram:\
        save_helper.get_waveform_from_normalized_spectogram(
            spectogram, [magnitude_stats, phase_stats], FFT_FRAME_LENGTH,
            FFT_FRAME_STEP, LOG_MAGNITUDE, INSTANTANEOUS_FREQUENCY
        )

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE, RESULT_DIR, get_waveform
        )

    spec_phase_gan_model = wgan.WGAN(
        normalized_raw_dataset, generator, [discriminator], Z_DIM,
        generator_optimizer, discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
        fn_save_examples=save_examples
    )

    spec_phase_gan_model.train()

if __name__ == '__main__':
    main()
