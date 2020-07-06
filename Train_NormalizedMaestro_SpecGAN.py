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
from librosa.core import griffinlim
from structures.SpecGAN import Generator, Discriminator
from models.WGAN import WGAN
from tensorflow.keras.activations import tanh
from datasets.MAESTRODataset import get_maestro_magnitude_phase_dataset
from utils.Spectral import magnitude_2_waveform
from tensorflow.keras.utils import Progbar

import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 64
BATCH_SIZE = 64
EPOCHS = 300

# Setup Dataset
maestro_path = 'data/MAESTRO_6h.npz'
raw_maestro = get_maestro_magnitude_phase_dataset(maestro_path, fft_length=256, frame_step=128)
raw_maestro = raw_maestro[:, :, :,0] # Remove the phase information

maestro_mean = np.mean(raw_maestro, axis=0)
maestro_std = np.std(raw_maestro, axis=0)

print(maestro_mean.shape)

def normalize(magnitude):
    norm = (magnitude - maestro_mean) / maestro_std
    norm /= 3.
    norm = tf.clip_by_value(norm, -1., 1.)
    return norm

normalized_raw_maestro = []
pb_i = Progbar(len(raw_maestro))
for d in raw_maestro:
    normalized_raw_maestro.append(normalize(d))
    pb_i.add(1)
    
#normalized_raw_maestro = np.array(normalized_raw_maestro)

# Construct generator and discriminator
generator = Generator(activation=tanh)
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = '_results/representation_study/SpecGAN_orig/training_checkpoints/'
#restore_point ='ckpt-5'

def get_waveform(magnitude):
    magnitude = np.reshape(magnitude, (128, 128))
    magnitude = magnitude * 3.
    magnitude = (magnitude * maestro_std) + maestro_mean
    
    return magnitude_2_waveform(magnitude, n_iter=16, fft_length=256, frame_step=128)

def save_examples(epoch, real, generated):
    gen_waveforms = []
    real_waveforms = []
    for r, g in zip(real, generated):
        real_waveforms.append(get_waveform(r))
        gen_waveforms.append(get_waveform(g))
        
    real_waveforms = np.reshape(real_waveforms, (-1))
    gen_waveforms = np.reshape(gen_waveforms, (-1))


    sf.write('_results/representation_study/SpecGAN_orig/audio/real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write('_results/representation_study/SpecGAN_orig/audio/gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

SpecGAN = WGAN(normalized_raw_maestro, [-1, 128, 128], [-1, 128, 128, 1], generator, discriminator, Z_dim, generator_optimizer, discriminator_optimizer, generator_training_ratio=D_updates_per_g, batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=checkpoint_dir, fn_save_examples=save_examples)

#SpecGAN.restore(restore_point, 50)
SpecGAN.train()
