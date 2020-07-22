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

"""Training Script for SpecGAN on MAESTRO.

This follows the origonal SpecGAN training,
where the magnitude spectrums are normalized
to sit between -1 and 1.
"""

import os
import tensorflow as tf
import numpy as np
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
GRIFFIN_LIM_ITERATIONS = 16
FFT_FRAME_LENGTH = 256
FFT_FRAME_STEP = 128
LOG_MAGNITUDE = True
SPECTOGRAM_IMAGE_SHAPE = [-1, FFT_FRAME_LENGTH // 2, FFT_FRAME_LENGTH // 2, 1]
CHECKPOINT_DIR = '_results/representation_study/SpecGAN/training_checkpoints/'
RESULT_DIR = '_results/representation_study/SpecGAN/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro, magnitude_stats, _ =\
        maestro_dataset.get_maestro_magnitude_phase_dataset(
            MAESTRO_PATH, FFT_FRAME_LENGTH, FFT_FRAME_STEP, LOG_MAGNITUDE
        )
    raw_maestro = raw_maestro[:, :, :, 0] # Remove the phase information

    normalized_raw_maestro = []
    pb_i = utils.Progbar(len(raw_maestro))
    for data_point in raw_maestro:
        normalized_raw_maestro.append(maestro_dataset.normalize(
            data_point, *magnitude_stats
        ))
        pb_i.add(1)
    normalized_raw_maestro = np.array(normalized_raw_maestro)

    generator = spec_gan.Generator(activation=activations.tanh)
    discriminator = spec_gan.Discriminator(input_shape=SPECTOGRAM_IMAGE_SHAPE)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda magnitude:\
        save_helper.get_waveform_from_normalized_magnitude(
            magnitude, magnitude_stats, GRIFFIN_LIM_ITERATIONS, FFT_FRAME_LENGTH,
            FFT_FRAME_STEP, LOG_MAGNITUDE
        )

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE, RESULT_DIR, get_waveform
        )

    spec_gan_model = wgan.WGAN(
        normalized_raw_maestro, generator, [discriminator], Z_DIM, generator_optimizer,
        discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
        fn_save_examples=save_examples
    )

    spec_gan_model.train()

if __name__ == '__main__':
    main()
