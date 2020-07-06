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
import structures.WaveGAN as WaveGAN
import structures.SpecGAN as SpecGAN
from datasets.MAESTRODataset import get_maestro_waveform_dataset, get_maestro_magnitude_phase_dataset
from utils.Spectral import waveform_2_spectogram, magnitude_2_waveform, spectogram_2_waveform
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import Progbar

import time
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

N_generations = 60
Z_dim = 64
results_path = '_results/representation_study/'
maestro_path = 'data/MAESTRO_6h.npz'

wavegan_checkpoint_path = '_results/representation_study/WaveGAN/training_checkpoints/ckpt-30'
specgan_checkpoint_path = '_results/representation_study/SpecGAN_orig/training_checkpoints/ckpt-23'
specphasegan_checkpoint_path = '_results/representation_study/SpecPhaseGAN/training_checkpoints/ckpt-8'

# Build and load models from checkpoints
# WaveGAN
wavegan_generator = WaveGAN.Generator()

wavegan_checkpoint = tf.train.Checkpoint(generator=wavegan_generator)
wavegan_checkpoint.restore(wavegan_checkpoint_path).expect_partial()


# SpecGAN
specgan_generator = SpecGAN.Generator(activation=tanh)

specgan_checkpoint = tf.train.Checkpoint(generator=specgan_generator)
specgan_checkpoint.restore(specgan_checkpoint_path).expect_partial()


# SpecPhaseGAN
specphasegan_generator = SpecGAN.Generator(channels=2, activation=tanh)

specphasegan_checkpoint = tf.train.Checkpoint(generator=specphasegan_generator)
specphasegan_checkpoint.restore(specphasegan_checkpoint_path).expect_partial()
##


# Generate Normalizing Coefficients:
# Needed for SpecGAN and SpecPhaseGAN
raw_maestro = get_maestro_magnitude_phase_dataset(maestro_path, fft_length=256, frame_step=128)

raw_maestro_magnitude = raw_maestro[:, :, :,0]
raw_maestro_phase = raw_maestro[:, :, :,1]

maestro_magnitude_mean = np.mean(raw_maestro_magnitude, axis=0)
maestro_magnitude_std = np.std(raw_maestro_magnitude, axis=0)

maestro_phase_mean = np.mean(raw_maestro_phase, axis=0)
maestro_phase_std = np.std(raw_maestro_phase, axis=0)
del raw_maestro
del raw_maestro_magnitude
del raw_maestro_phase

def un_normalize_magnitude(magnitude):
    """Un-normalize a given (normalized) magnitude spectrum.
    """
    magnitude = np.reshape(magnitude, (128, 128))
    magnitude = magnitude * 3.
    magnitude = (magnitude * maestro_magnitude_std) + maestro_magnitude_mean
    return magnitude
    
def un_normalize_spectogram(spectogram):
    """Un-normalize a given (normalized) spectogram.
    """
    spectogram = np.reshape(spectogram, (128, 128, 2))
    
    magnitude = np.reshape(spectogram[:,:,0], (128, 128))
    magnitude = magnitude * 3.
    magnitude = (magnitude * maestro_magnitude_std) + maestro_magnitude_mean
    
    phase = np.reshape(spectogram[:,:,1], (128, 128))
    phase = phase * 3.
    phase = (phase * maestro_phase_std) + maestro_phase_mean
    
    spectogram = np.concatenate([np.expand_dims(magnitude, axis=2), np.expand_dims(phase, axis=2)], axis=-1)
    return spectogram


maestro = get_maestro_waveform_dataset(maestro_path)
maestro = maestro[np.random.randint(low=0, high=len(maestro), size=(N_generations))]
z = tf.random.uniform(shape=(N_generations, Z_dim), minval=-1, maxval=1)

# Produce generations for each component model, convert
# them to waveforms and record them
real_data = {'waveform': [], 'waveform_gl': [], 'waveform_istft': []}
wavegan_generations = {'waveform': []}
specgan_generations = {'waveform': []}
specphasegan_generations = {'waveform': []}

pb_i = Progbar(N_generations)
for i in range(N_generations):
    pb_i.add(1)
    # Process real data
    real_data['waveform'].append(maestro[i])
    real_data_spectogram = waveform_2_spectogram(maestro[i], fft_length=256, frame_step=128)
    real_data['waveform_gl'].append(magnitude_2_waveform(real_data_spectogram[:,:,0], fft_length=256, frame_step=128)[0:2**14])
    real_data['waveform_istft'].append(spectogram_2_waveform(real_data_spectogram, fft_length=256, frame_step=128)[0:2**14])
    
    # Process WaveGAN
    wavegan_waveform = wavegan_generator(tf.reshape(z[i], (1, Z_dim)))
    wavegan_generations['waveform'].append(tf.reshape(wavegan_waveform, (-1)))
    
    # Process SpecGAN
    specgan_magnitude = specgan_generator(tf.reshape(z[i], (1, Z_dim)))
    specgan_magnitude = un_normalize_magnitude(tf.reshape(specgan_magnitude, (128,128)))
    specgan_waveform = magnitude_2_waveform(specgan_magnitude, fft_length=256, frame_step=128)[0:2**14]
    print(specgan_waveform.shape)
    specgan_generations['waveform'].append(specgan_waveform)
    
    # Process SpecPhaseGAN
    specphasegan_spectogram = specphasegan_generator(tf.reshape(z[i], (1, Z_dim)))
    specphasegan_spectogram = un_normalize_spectogram(tf.reshape(specphasegan_spectogram, (128,128, 2)))
    specphasegan_generations['waveform'].append(spectogram_2_waveform(specphasegan_spectogram, fft_length=256, frame_step=128)[0:2**14])
    

    
# Save Real Data
sf.write(results_path + 'Real.wav', np.reshape(real_data['waveform'], (-1)), 16000)
sf.write(results_path + 'Real_Griffin_Lim.wav', np.reshape(real_data['waveform_gl'], (-1)), 16000)
sf.write(results_path + 'Real_iSTFT.wav', np.reshape(real_data['waveform_istft'], (-1)), 16000)
print('Saved Real Data')

# Save WGAN Data
sf.write(results_path + 'WaveGAN.wav', np.reshape(wavegan_generations['waveform'], (-1)), 16000)
print('Saved WaveGAN Data')

# Save SpecGAN Data
sf.write(results_path + 'SpecGAN.wav', np.reshape(specgan_generations['waveform'], (-1)), 16000)
print('Saved SpecGAN Data')

# Save SpecPhaseGAN Data
sf.write(results_path + 'SpecPhaseGAN.wav', np.reshape(specphasegan_generations['waveform'], (-1)), 16000)
print('Saved SpecPhaseGAN Data')