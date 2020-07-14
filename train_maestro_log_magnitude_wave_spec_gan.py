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

import tensorflow as tf
from tensorflow.keras import utils
from audio_synthesis.structures import log_spec_conditional_wave_gan as wave_gan
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models.conditional_wgan import WGAN
from audio_synthesis.utils import spectral
import time
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
CHECKPOINT_DIR = '_results/music_coding/log_magnitude_conditioning/training_checkpoints/'
RESULT_DIR = '_results/music_coding/log_magnitude_conditioning/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)

raw_conditional_maestro = []
pb_i = utils.Progbar(len(raw_maestro))
for data_point in raw_maestro:
    conditioning = spectral.waveform_2_magnitude(data_point, 256, 128, True)
    raw_conditional_maestro.append(conditioning)
    
    pb_i.add(1)
raw_conditional_maestro = np.array(raw_conditional_maestro)

generator = wave_gan.Generator()
discriminator = wave_gan.ConditionalDiscriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

def save_examples(epoch, real, generated):  
    real_waveforms = np.reshape(real, (-1))
    gen_waveforms = np.reshape(generated, (-1))

    sf.write(RESULT_DIR + 'real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write(RESULT_DIR + 'gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

    
WaveGAN = WGAN((raw_maestro, raw_conditional_maestro), [[-1, 2**14, 1], [-1, 128, 128, 1]], generator, discriminator, Z_DIM, generator_optimizer, discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G, batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR, fn_save_examples=save_examples)

WaveGAN.train()