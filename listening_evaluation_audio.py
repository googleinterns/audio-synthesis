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
import errno
from tensorflow.keras import activations
from tensorflow.keras import utils
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral
from audio_synthesis.structures import wave_gan, spec_gan

N_GENERATIONS = 1
GENERATION_LENGTH = 5
Z_DIM = 64
WAVEFORM_LENGTH = 16000
SAMPLING_RATE = 16000
RESULTS_PATH = '_results/representation_study/listening_test/MAESTRO/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'
GRIFFIN_LIM_ITERATIONS = 32
FFT_FRAME_LENGTHS = [512, 512]
FFT_FRAME_STEPS = [128, 128]
LOG_MAGNITUDE = [True, True]
INSTANTANEOUS_FREQUENCY = [True, False]

def mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def _data_waveform_griffin_lim_fn(data_waveform, frame_length, frame_step):
    """Converts a waveform to a magnitude spectrum and
    then back to a waveform using Griffin-Lim.

    Args:
        data_waveform: Initial waveform.

    Returns:
        Waveform with same shape as input, but with phase
        estimated using Griffin-Lim.
    """

    data_spectogram = spectral.waveform_2_spectogram(
        data_waveform,
        frame_length=frame_length,
        frame_step=frame_step
    )
    return spectral.magnitude_2_waveform(
        data_spectogram[:, :, :, 0],
        frame_length=frame_length,
        frame_step=frame_step
    )

# An object containing the models we wish to process and extract results from.
# There are two options:
#   1) Either the model is a trained generator function. In this case,
#      a 'generator' model must be specified, along with a 'checkpoint_path'
#      to load from. In addition, a 'preprocess' object must be specified,
#      that contains 'unnormalize_magnitude' and 'unnormalize_spectogram' (mutually exclusive)
#   2) Or it is a processed form of the origonal data. In this case, 'data': True
#      must be set.
# All models must have 'generate_fn' set to a function that takes a generation from
# that mode (or data point) and returns a waveform. Additionally, waveform': [] must
# set, this is where the waveforms are collected.
MODELS = {
    'WaveGAN': {
        'generator': wave_gan.Generator(),
        'checkpoint_path':\
            '_results/representation_study/WaveGAN/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': False,
        },
        'generate_fn': lambda x: x,
        'waveform': [],
    },
    'STFTGAN_HR': {
        'generator': spec_gan.Generator(channels=2, in_shape=[4, 8, 1024]),
        'checkpoint_path':\
            '_results/representation_study/STFTGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': False,
        },
        'fft_config': 0,
        'generate_fn': lambda stfts: spectral.stft_2_waveform(
            stfts, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1]
        ),
        'waveform': [],
    },
    'STFTWaveGAN_HR': {
        'generator': spec_gan.Generator(channels=2, in_shape=[4, 8, 1024]),
        'checkpoint_path':\
            '_results/representation_study/STFTWaveGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': False,
        },
        'fft_config': 0,
        'generate_fn': lambda stfts: spectral.stft_2_waveform(
            stfts, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1]
        ),
        'waveform': [],
    },
    'STFTMagGAN_HR': {
        'generator': spec_gan.Generator(channels=2, in_shape=[4, 8, 1024]),
        'checkpoint_path':\
            '_results/representation_study/STFTSpecGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': False,
        },
        'fft_config': 0,
        'generate_fn': lambda stfts: spectral.stft_2_waveform(
            stfts, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1]
        ),
        'waveform': [],
    },
    'MagGAN_HR': {
        'generator': spec_gan.Generator(activation=activations.tanh, in_shape=[4, 8, 1024]),
        'checkpoint_path':\
            '_results/representation_study/SpecGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': True,
            'unnormalize_spectogram': False,
        },
        'fft_config': 0,
        'generate_fn': lambda magnitude: spectral.magnitude_2_waveform(
            magnitude, GRIFFIN_LIM_ITERATIONS, FFT_FRAME_LENGTHS[1],
            FFT_FRAME_STEPS[1], LOG_MAGNITUDE
        ),
        'waveform': [],
    },
    'MagPhaseGAN_HR': {
        'generator': spec_gan.Generator(
            channels=2, activation=activations.tanh, in_shape=[4, 8, 1024]
        ),
        'checkpoint_path':\
            '_results/representation_study/MagPhaseGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': True,
        },
        'fft_config': 1,
        'generate_fn': lambda spectogram: spectral.spectogram_2_waveform(
            spectogram, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1], LOG_MAGNITUDE[0],
            INSTANTANEOUS_FREQUENCY[0]
        ),
        'waveform': [],
    },
    'MagIFGAN_HR': {
        'generator': spec_gan.Generator(
            channels=2, activation=activations.tanh, in_shape=[4, 8, 1024]
        ),
        'checkpoint_path':\
            '_results/representation_study/SpecPhaseGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': True,
        },
        'fft_config': 1,
        'generate_fn': lambda spectogram: spectral.spectogram_2_waveform(
            spectogram, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1], LOG_MAGNITUDE[1],
            INSTANTANEOUS_FREQUENCY[1]
        ),
        'waveform': [],
    },
    'WaveMagGAN_HR': {
        'generator': wave_gan.Generator(),
        'checkpoint_path':\
            '_results/representation_study/WaveSpecGAN_HR/training_checkpoints/ckpt-30',
        'preprocess': {
            'unnormalize_magnitude': False,
            'unnormalize_spectogram': False,
        },
        'generate_fn': lambda x: x,
        'waveform': [],
    },
    'Waveform': {
        'data': True,
        'generate_fn': lambda x: x,
        'waveform': [],
    },
    'Waveform_GL_HR': {
        'data': True,
        'generate_fn': lambda waveform: _data_waveform_griffin_lim_fn(
            waveform, FFT_FRAME_LENGTHS[1], FFT_FRAME_STEPS[1]
        ),
        'waveform': [],
    },
}

def main():
    # Set allowed GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Build and load MODELS from checkpoints
    for model_name in MODELS:
        MODELS[model_name]['loaded'] = True
        if 'data' in MODELS[model_name] and MODELS[model_name]['data']:
            continue

        try:
            checkpoint = tf.train.Checkpoint(generator=MODELS[model_name]['generator'])
            checkpoint.restore(MODELS[model_name]['checkpoint_path']).expect_partial()
            print('Loaded ', model_name)
        except:
            print(model_name, ' not found')
            MODELS[model_name]['loaded'] = False

    maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)

    magnitude_stastics = []
    phase_stastics = []
    for frame_length, frame_step in zip(FFT_FRAME_LENGTHS, FFT_FRAME_STEPS):
        _, magnitude_stastic, phase_stastic =\
            maestro_dataset.get_maestro_magnitude_phase_dataset(
                MAESTRO_PATH, frame_length, frame_step, LOG_MAGNITUDE,
                INSTANTANEOUS_FREQUENCY
            )
        
        magnitude_stastics.append(magnitude_stastic)
        phase_stastics.append(phase_stastic)

    maestro = maestro[np.random.randint(low=0, high=len(maestro), size=N_GENERATIONS)]
    z_gen = tf.random.uniform((N_GENERATIONS, GENERATION_LENGTH, Z_DIM), -1, 1, tf.float32)

    pb_i = utils.Progbar(N_GENERATIONS)
    for i in range(N_GENERATIONS):
        z_in = tf.reshape(z_gen[i], (GENERATION_LENGTH, Z_DIM))
        print(z_in.shape)

        for model_name in MODELS:
            print(model_name)
            if not MODELS[model_name]['loaded']:
                continue
            
            # If the model is a generator then produce a random generation,
            # otherwise take the current data point.
            if 'data' in MODELS[model_name] and MODELS[model_name]['data']:
                generation = maestro[i]
            else:
                generation = MODELS[model_name]['generator'](z_in)
                generation = np.squeeze(generation)

                if MODELS[model_name]['preprocess']['unnormalize_magnitude']:
                    fft_config = MODELS[model_name]['fft_config']
                    generation = maestro_dataset.un_normalize(
                        generation, *magnitude_stastics[fft_config]
                    )
                elif MODELS[model_name]['preprocess']['unnormalize_spectogram']:
                    fft_config = MODELS[model_name]['fft_config']
                    generation = maestro_dataset.un_normalize_spectogram(
                        generation, magnitude_stastics[fft_config], phase_stastics[fft_config]
                    )

            # Apply pre-defined transform to waveform.
            generation = np.squeeze(generation)
            waveform = MODELS[model_name]['generate_fn'](generation)
            print(waveform.shape)

            # Clip waveform to desired length and save
            waveform = waveform[:, 0:WAVEFORM_LENGTH]
            MODELS[model_name]['waveform'].append(waveform)

        pb_i.add(1)

    # Save the waveforms for each model as one long audio clip
    for model_name in MODELS:
        if not MODELS[model_name]['loaded']:
            continue
        
        path = os.join(RESULTS_PATH, model_name)
        mkdir(path)
        for i, generation in enumerate(MODELS[model_name]['waveform']):
            generation = np.array(generation)
            print(generation.shape)
            wav = np.reshape(MODELS[model_name]['waveform'], (-1))
            sf.write(os.path.join(path, model_name + '_{}.wav'.format(i)), wav, SAMPLING_RATE)

if __name__ == '__main__':
    main()
