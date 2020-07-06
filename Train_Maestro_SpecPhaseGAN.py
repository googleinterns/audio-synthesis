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
from structures.SpecGAN import Generator, Discriminator
from models.WGAN import WGAN
from datasets.MAESTRODataset import get_maestro_magnitude_phase_dataset
from utils.Spectral import spectogram_2_waveform, magnitude_2_waveform
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import Progbar

import time
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 64
BATCH_SIZE = 64
EPOCHS = 300

# Setup Dataset
maestro_path = 'data/MAESTRO_6h.npz'
raw_maestro = get_maestro_magnitude_phase_dataset(maestro_path, fft_length=256, frame_step=128)

raw_maestro_magnitude = raw_maestro[:, :, :,0]
raw_maestro_phase = raw_maestro[:, :, :,1]

maestro_magnitude_mean = np.mean(raw_maestro_magnitude, axis=0)
maestro_magnitude_std = np.std(raw_maestro_magnitude, axis=0)

maestro_phase_mean = np.mean(raw_maestro_phase, axis=0)
maestro_phase_std = np.std(raw_maestro_phase, axis=0)

print(maestro_magnitude_mean.shape)

def normalize(magnitude, mean, std):
    norm = (magnitude - mean) / std
    norm /= 3.
    norm = tf.clip_by_value(norm, -1., 1.)
    return norm


normalized_raw_maestro = []
pb_i = Progbar(len(raw_maestro))
for i in range(len(raw_maestro)):
    norm_mag = normalize(raw_maestro_magnitude[i], maestro_magnitude_mean, maestro_magnitude_std)
    norm_phase = normalize(raw_maestro_phase[i], maestro_phase_mean, maestro_phase_std)
    
    norm = np.concatenate([np.expand_dims(norm_mag, axis=2), np.expand_dims(norm_phase, axis=2)], axis=-1)
    normalized_raw_maestro.append(norm)
    pb_i.add(1)

# Construct generator and discriminator
generator = Generator(channels=2, activation=tanh)
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = '_results/representation_study/SpecPhaseGAN/training_checkpoints/'
#restore_point ='ckpt-7'



def get_waveform(spectogram):
    spectogram = np.reshape(spectogram, (128, 128, 2))
    
    magnitude = np.reshape(spectogram[:,:,0], (128, 128))
    magnitude = magnitude * 3.
    magnitude = (magnitude * maestro_magnitude_std) + maestro_magnitude_mean
    
    phase = np.reshape(spectogram[:,:,1], (128, 128))
    phase = phase * 3.
    phase = (phase * maestro_phase_std) + maestro_phase_mean
    
    spectogram = np.concatenate([np.expand_dims(magnitude, axis=2), np.expand_dims(phase, axis=2)], axis=-1)
    
    return spectogram_2_waveform(spectogram, fft_length=256, frame_step=128, log_magnitude=True, instantaneous_frequency=True)

def get_waveform_gl(magnitude):
    magnitude = np.reshape(magnitude, (128, 128))
    magnitude = magnitude * 3.
    magnitude = (magnitude * maestro_magnitude_std) + maestro_magnitude_mean
    
    return magnitude_2_waveform(magnitude, n_iter=16, fft_length=256, frame_step=128)

def save_examples(epoch, real, generated):
    gen_waveforms = []
    gen_griffin_lim_waveforms = []
    real_waveforms = []
    for r, g in zip(real, generated):
        real_waveforms.append(get_waveform(r))
        gen_waveforms.append(get_waveform(g))
        gen_griffin_lim_waveforms.append(get_waveform_gl(g[:,:,0]))
        
        
    real_waveforms = np.reshape(real_waveforms, (-1))
    gen_waveforms = np.reshape(gen_waveforms, (-1))
    gen_griffin_lim_waveforms = np.reshape(gen_griffin_lim_waveforms, (-1))


    sf.write('_results/representation_study/SpecPhaseGAN/audio/real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write('_results/representation_study/SpecPhaseGAN/audio/gen_' + str(epoch) + '.wav', gen_waveforms, 16000)
    sf.write('_results/representation_study/SpecPhaseGAN/audio/gen_gl_' + str(epoch) + '.wav', gen_griffin_lim_waveforms, 16000)

SpecPhaseGAN = WGAN(normalized_raw_maestro, [-1, 128, 128, 2], [-1, 128, 128, 2], generator, discriminator, Z_dim, generator_optimizer, discriminator_optimizer, generator_training_ratio=D_updates_per_g, batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=checkpoint_dir, fn_save_examples=save_examples)

#SpecPhaseGAN.restore(restore_point, 70)
SpecPhaseGAN.train()
