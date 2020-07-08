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

"""Training Script for WaveGAN on NSynth
"""

import os
import soundfile as sf
import numpy as np
import tensorflow as tf
from audio_synthesis.models.wgan import WGAN
from audio_synthesis.structures.wave_gan import Generator, Discriminator
from audio_synthesis.third_party.NSynthDataset import NSynthTFRecordDataset


os.environ["CUDA_VISIBLE_DEVICES"] = ''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 32
BATCH_SIZE = 16
EPOCHS = 100
NSYNTH_PATH = '../data/nsynth-train.tfrecord'
CHECKPOINT_DIR = None
RESULTS_PATH = '_results/rwd/WaveGAN/audio/'


nsynth = NSynthTFRecordDataset(NSYNTH_PATH).provide_dataset()
raw_nsynth = np.array(nsynth.as_numpy_iterator())

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

def save_examples(epoch, real, generated):
    """Saves a batch of real and generated data
    """

    real_waveforms = np.reshape(real, (-1))
    gen_waveforms = np.reshape(generated, (-1))

    sf.write(RESULTS_PATH + 'real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write(RESULTS_PATH + 'gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

WaveGAN = WGAN(raw_nsynth, [-1, 2**16], [-1, 2**16, 1], generator,
               discriminator, Z_DIM, generator_optimizer, discriminator_optimizer,
               generator_training_ratio=D_UPDATES_PER_G, batch_size=BATCH_SIZE,
               epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR,
               fn_save_examples=save_examples)

WaveGAN.train()
