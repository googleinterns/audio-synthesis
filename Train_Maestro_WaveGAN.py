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
from structures.WaveGAN import Generator, Discriminator
from datasets.MAESTRODataset import get_maestro_waveform_dataset
from models.WGAN import WGAN
import time
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 64
BATCH_SIZE = 64
EPOCHS = 3000

# Setup Dataset
maestro_path = 'data/MAESTRO_6h.npz'
raw_maestro = get_maestro_waveform_dataset(maestro_path)

# Construct generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = '_results/representation_study/WaveGAN/training_checkpoints/'

def save_examples(epoch, real, generated):  
    real_waveforms = np.reshape(real, (-1))
    gen_waveforms = np.reshape(generated, (-1))

    sf.write('_results/representation_study/SpecGAN/audio/real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write('_results/representation_study/SpecGAN/audio/gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

    
WaveGAN = WGAN(raw_maestro, [-1, 2**14], [-1, 2**14, 1], generator, discriminator, Z_dim, generator_optimizer, discriminator_optimizer, generator_training_ratio=D_updates_per_g, batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=checkpoint_dir, fn_save_examples=save_examples)

WaveGAN.train()