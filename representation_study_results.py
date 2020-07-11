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

"""This module handles generating audio from models trained
in the representation study. The list of models is modular.
"""

import os
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import Progbar
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.datasets.maestro_dataset import get_maestro_waveform_dataset,\
        get_maestro_spectogram_normalizing_constants, un_normalize
from audio_synthesis.utils.spectral import waveform_2_spectogram,\
        magnitude_2_waveform, spectogram_2_waveform
import audio_synthesis.structures.wave_gan as WaveGAN
import audio_synthesis.structures.spec_gan as SpecGAN

os.environ["CUDA_VISIBLE_DEVICES"] = ''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

N_GENERATIONS = 60
Z_DIM = 64
WAVEFORM_LENGTH = 16000
SAMPLING_RATE = 16000
RESULTS_PATH = '_results/representation_study/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'
FFT_FRAME_LENGTH = 256
FFT_FRAME_STEP = 128

maestro_magnitude_mean, maestro_magnitude_std, maestro_phase_mean, maestro_phase_std = \
            get_maestro_spectogram_normalizing_constants(MAESTRO_PATH,
                                                         frame_length=FFT_FRAME_LENGTH,
                                                         frame_step=FFT_FRAME_STEP)

def un_normalize_magnitude(magnitude):
    """Un-normalize a given (normalized) magnitude spectrum
    """
    return un_normalize(np.squeeze(magnitude),
                        maestro_magnitude_mean, maestro_magnitude_std)

def un_normalize_spectogram(spectogram):
    """Un-normalize a given (normalized) spectogram.
    """

    magnitude = un_normalize_magnitude(np.squeeze(spectogram[:, :, 0]))
    phase = un_normalize(np.squeeze(spectogram[:, :, 1]),
                         maestro_phase_mean, maestro_phase_std)

    spectogram = np.concatenate([np.expand_dims(magnitude, axis=2),
                                 np.expand_dims(phase, axis=2)], axis=-1)
    return spectogram


# For each model, we must define how to convert a single generation
# into a waveform. They are given a single generation from the model
# in question and must return a waveform representation.
def _wave_gan_generate_fn(generated_waveform):
    return generated_waveform

def _spec_gan_generate_fn(generated_magnitude):
    generated_magnitude = un_normalize_magnitude(generated_magnitude)
    return magnitude_2_waveform(generated_magnitude,
                                frame_length=FFT_FRAME_LENGTH,
                                frame_step=FFT_FRAME_STEP)

def _spec_phase_gan_generate_fn(generated_spectogram):
    generated_spectogram = un_normalize_spectogram(generated_spectogram)
    return spectogram_2_waveform(generated_spectogram,
                                 frame_length=FFT_FRAME_LENGTH,
                                 frame_step=FFT_FRAME_STEP)

def _data_waveform_fn(data_waveform):
    return data_waveform

def _data_waveform_gl_fn(data_waveform):
    data_spectogram = waveform_2_spectogram(data_waveform,
                                            frame_length=FFT_FRAME_LENGTH,
                                            frame_step=FFT_FRAME_STEP)
    return magnitude_2_waveform(data_spectogram[:, :, 0],
                                frame_length=FFT_FRAME_LENGTH,
                                frame_step=FFT_FRAME_STEP)

def _data_waveform_istft_fn(data_waveform):
    data_spectogram = waveform_2_spectogram(data_waveform,
                                            frame_length=FFT_FRAME_LENGTH,
                                            frame_step=FFT_FRAME_STEP)
    return spectogram_2_waveform(data_spectogram,
                                 frame_length=FFT_FRAME_LENGTH,
                                 frame_step=FFT_FRAME_STEP)

# An object containing the models we wish to process and extract results from.
# There are two options:
#   1) Either the model is a trained generator function. In this case,
#      a 'generator' model must be specified, along with a 'checkpoint_path'
#      to load from.
#   2) Or it is a processed form of the origonal data. In this case, 'data': True
#      must be set.
# All models must have 'generate_fn' set to a function that takes a generation from
# that mode (or data point) and returns a waveform. Additionally, waveform': [] must
# set, this is where the waveforms are collected.
models = {
    'WaveGAN': {
        'generator': WaveGAN.Generator(),
        'checkpoint_path':\
            '_results/representation_study/WaveGAN/training_checkpoints/ckpt-30',
        'generate_fn': _wave_gan_generate_fn,
        'waveform': [],
    },
    'SpecGAN': {
        'generator': SpecGAN.Generator(activation=tanh),
        'checkpoint_path':\
            '_results/representation_study/SpecGAN_orig/training_checkpoints/ckpt-30',
        'generate_fn': _spec_gan_generate_fn,
        'waveform': [],
    },
    'SpecPhaseGAN': {
        'generator': SpecGAN.Generator(channels=2, activation=tanh),
        'checkpoint_path':\
            '_results/representation_study/SpecPhaseGAN/training_checkpoints/ckpt-30',
        'generate_fn': _spec_phase_gan_generate_fn,
        'waveform': [],
    },
    'WaveSpecGAN': {
        'generator': WaveGAN.Generator(),
        'checkpoint_path':\
            '_results/representation_study/WaveSpecGAN/training_checkpoints/ckpt-15',
        'generate_fn': _wave_gan_generate_fn,
        'waveform': [],
    },
    'Waveform': {
        'data': True,
        'generate_fn': _data_waveform_fn,
        'waveform': [],
    },
    'Waveform_GL': {
        'data': True,
        'generate_fn': _data_waveform_gl_fn,
        'waveform': [],
    },
    'Waveform_iSTFT': {
        'data': True,
        'generate_fn': _data_waveform_istft_fn,
        'waveform': [],
    }
}


if __name__ == '__main__':
    # Build and load models from checkpoints
    for model_name in models:
        if 'data' in models[model_name] and models[model_name]['data']:
            continue

        checkpoint = tf.train.Checkpoint(generator=models[model_name]['generator'])
        checkpoint.restore(models[model_name]['checkpoint_path']).expect_partial()


    maestro = get_maestro_waveform_dataset(MAESTRO_PATH)
    maestro = maestro[np.random.randint(low=0, high=len(maestro), size=(N_GENERATIONS))]
    z = tf.random.uniform((N_GENERATIONS, Z_DIM), -1, 1, 'float32')

    pb_i = Progbar(N_GENERATIONS)
    for i in range(N_GENERATIONS):
        z_in = tf.reshape(z[i], (1, Z_DIM))

        for model_name in models:
            # If the model is a generator then produce a random generation,
            # otherwise take the current data point.
            if 'data' in models[model_name] and models[model_name]['data']:
                generation = maestro[i]
            else:
                generation = models[model_name]['generator'](z_in)

            # Apply pre-defined transform to waveform.
            generation = np.squeeze(generation)
            waveform = models[model_name]['generate_fn'](generation)

            # Clip waveform to desired length and save
            waveform = waveform[0:WAVEFORM_LENGTH]
            models[model_name]['waveform'].append(waveform)

        pb_i.add(1)

    # Save the waveforms for each model as one long audio clip
    for model_name in models:
        wav = np.reshape(models[model_name]['waveform'], (-1))
        sf.write(RESULTS_PATH + model_name + '.wav', wav, SAMPLING_RATE)
