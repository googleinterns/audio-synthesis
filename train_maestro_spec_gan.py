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

"""Training Script for SpecGAN on MAESTRO
"""

import os
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import Progbar
from audio_synthesis.structures.spec_gan import Generator, Discriminator
from audio_synthesis.models.wgan import WGAN
from audio_synthesis.datasets.maestro_dataset import get_maestro_magnitude_phase_dataset,\
        get_maestro_spectogram_normalizing_constants, normalize, un_normalize
from audio_synthesis.utils.spectral import magnitude_2_waveform

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 256
FFT_FRAME_STEP = 128
CHECKPOINT_DIR = '_results/representation_study/SpecGAN/training_checkpoints/'
RESULT_DIR = '_results/representation_study/SpecGAN/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

# Setup Dataset
raw_maestro = get_maestro_magnitude_phase_dataset(MAESTRO_PATH, FFT_FRAME_LENGTH, FFT_FRAME_STEP)
raw_maestro = raw_maestro[:, :, :, 0] # Remove the phase information

maestro_magnitude_mean, maestro_magnitude_std, _, _ =\
        get_maestro_spectogram_normalizing_constants(MAESTRO_PATH,
                                                     FFT_FRAME_LENGTH,
                                                     FFT_FRAME_STEP)

normalized_raw_maestro = []
pb_i = Progbar(len(raw_maestro))
for d in raw_maestro:
    normalized_raw_maestro.append(normalize(d, maestro_magnitude_mean,
                                            maestro_magnitude_std))
    pb_i.add(1)

generator = Generator(activation=tanh)
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

def get_waveform(magnitude):
    """A wrapper for the 'magnitude_2_waveform' function
    that handles un-normalization.
    """

    magnitude = np.squeeze(magnitude)
    magnitude = un_normalize(magnitude, maestro_magnitude_mean,
                             maestro_magnitude_std)

    return magnitude_2_waveform(magnitude, n_iter=16,
                                frame_length=FFT_FRAME_LENGTH,
                                frame_step=FFT_FRAME_STEP)

def save_examples(epoch, real, generated):
    """Saves a batch of real and generated data.
    """

    gen_waveforms = []
    real_waveforms = []
    for real_magnitude, generated_magnitude in zip(real, generated):
        real_waveforms.append(get_waveform(real_magnitude))
        gen_waveforms.append(get_waveform(generated_magnitude))

    real_waveforms = np.reshape(real_waveforms, (-1))
    gen_waveforms = np.reshape(gen_waveforms, (-1))

    sf.write(RESULT_DIR + 'real_' + str(epoch) + '.wav', real_waveforms, SAMPLING_RATE)
    sf.write(RESULT_DIR + 'gen_' + str(epoch) + '.wav', gen_waveforms, SAMPLING_RATE)

SpecGAN = WGAN(normalized_raw_maestro, [-1, 128, 128], [-1, 128, 128, 1], generator,
               discriminator, Z_DIM, generator_optimizer, discriminator_optimizer,
               generator_training_ratio=D_UPDATES_PER_G, batch_size=BATCH_SIZE,
               epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
               fn_save_examples=save_examples)

SpecGAN.train()
