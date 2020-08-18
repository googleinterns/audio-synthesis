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

"""
"""

import sys
import os
import errno
import tensorflow as tf
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pesq
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral

def mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def add_noise_at_snr_channel(channel_in, snr):
    RMS = np.sqrt(np.mean(channel_in ** 2))
    RMS_noise = RMS / np.sqrt(10 ** (snr/10))
    noise_std = RMS_noise

    return channel_in + np.random.normal(size=channel_in.shape, scale=noise_std)

def distort_two_channel_representation(representation, snr):
    distorted_channels = []
    orig_channels = []
    for channel in range(representation.shape[-1]):
        input_channel = representation[:, :, channel]
        orig_channels.append(np.expand_dims(input_channel, 2))
        distorted_channel = add_noise_at_snr_channel(input_channel, snr)
        distorted_channels.append(np.expand_dims(distorted_channel, 2))

    magnitude_only = np.concatenate([distorted_channels[0], orig_channels[1]], axis=-1)
    phase_only = np.concatenate([orig_channels[0], distorted_channels[1]], axis=-1)
    return magnitude_only, phase_only
    
def distort_one_channel_representation(representation, snr):
    return np.expand_dims(add_noise_at_snr_channel(representation, snr), 0)

def perceptual_error(audio_hat, audio): 
    audio_hat = np.array(audio_hat)
    audio = np.array(audio)
    return pesq.pesq(SAMPLE_RATE, audio, audio_hat, 'wb')

MAX_SNR = 80
MIN_SNR = 0
SNR_STEP = 2
SNRS = list(range(MIN_SNR, MAX_SNR+SNR_STEP, SNR_STEP))
SAMPLE_RATE = 16000
LOG_MAGNITUDE = False
DATASET_PATH = './data/SpeechMNIST_1850.npz'
RESULTS_PATH = './_results/snr_vs_pesq/'
LINE_STYLES = ['--', ':']

REPRESENTATIONS = {
    'STFT': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step: spectral.waveform_2_stft(
            waveform, window_length, window_step
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step: spectral.stft_2_waveform(
            representation, window_length, window_step
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Mag': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step: spectral.waveform_2_magnitude(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step: spectral.magnitude_2_waveform(
            representation, 32, window_length, window_step, log_magnitude=LOG_MAGNITUDE
        )[0],
        'distort_representation': distort_one_channel_representation,
    },
    'Mag + Phase': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step: spectral.waveform_2_spectogram(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=False
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step: spectral.spectogram_2_waveform(
            representation, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=False
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Mag + IF': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step: spectral.waveform_2_spectogram(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=True
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step: spectral.spectogram_2_waveform(
            representation, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=True
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Waveform': {
        'requires_fft_params': False,
        'waveform_2_representation': lambda x: x,
        'representation_2_waveform': lambda x: x,
        'distort_representation': distort_one_channel_representation,
    }
}

def main(fft_window_size, fft_window_step):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(DATASET_PATH)
    origonal_audio = np.array(raw_maestro[0]).astype(np.float32)
    
    for representation in REPRESENTATIONS:
        REPRESENTATIONS[representation]['perceptual_errors'] = []
        REPRESENTATIONS[representation]['waveforms'] = []
    
    for snr in SNRS:
        for representation in REPRESENTATIONS:
            if REPRESENTATIONS[representation]['requires_fft_params']:
                audio_representation = REPRESENTATIONS[representation]['waveform_2_representation'](
                    origonal_audio, fft_window_size, fft_window_step
                )
            else:
                audio_representation = REPRESENTATIONS[representation]['waveform_2_representation'](origonal_audio)
                
            noisy_representations = REPRESENTATIONS[representation]['distort_representation'](audio_representation, snr)
            
            perceptual_errors = []
            audio_hats = []
            for noisy_representation in noisy_representations:
                if REPRESENTATIONS[representation]['requires_fft_params']:
                    audio_hat = REPRESENTATIONS[representation]['representation_2_waveform'](
                        noisy_representation, fft_window_size, fft_window_step
                    )
                else:
                    audio_hat = REPRESENTATIONS[representation]['representation_2_waveform'](noisy_representation)
                
                audio_hat = audio_hat[0:len(origonal_audio)]
                
                audio_hats.append(audio_hat)
                perceptual_errors.append(perceptual_error(origonal_audio, audio_hat))
                
            
            REPRESENTATIONS[representation]['perceptual_errors'].append(perceptual_errors)
            REPRESENTATIONS[representation]['waveforms'].append(audio_hats)
            
    
    # Plot the graph
    for representation in REPRESENTATIONS:
        perceptual_errors = REPRESENTATIONS[representation]['perceptual_errors']
        perceptual_errors = np.array(perceptual_errors)
        
        color = None
        label = representation
        
        p = plt.plot(SNRS, perceptual_errors[:,0], label=representation)
        
        for i in range(perceptual_errors.shape[-1] - 1):
            plt.plot(SNRS, perceptual_errors[:,i+1], color=p[0].get_color(), linestyle=LINE_STYLES[i])
            
    plt.xlabel('SNR')
    plt.ylabel('PESQ')
    plt.legend()
    
    file_name = 'pesq_vs_snr__{}ws_{}s'.format(fft_window_size, fft_window_step)
    plt.savefig(os.path.join(RESULTS_PATH, file_name))
    plt.clf()
        
    # Save the audio files
    setup = 'audio_{}ws_{}s'.format(fft_window_size, fft_window_step)
    base_audio_dir = os.path.join(RESULTS_PATH, setup)
    mkdir(base_audio_dir)
    for representation in REPRESENTATIONS:
        audio_dir = os.path.join(base_audio_dir, representation)
        mkdir(audio_dir)
        for i, audio in enumerate(REPRESENTATIONS[representation]['waveforms']):
            for j, channel in enumerate(audio):
                file_path = os.path.join(audio_dir, '{}_{}db_{}.wav'.format(representation, SNRS[i], j))
                sf.write(file_path, audio[j], SAMPLE_RATE)
                
    
if __name__ == '__main__':
    main(512, 64)
    main(512, 128)
    main(512, 256)
    main(256, 128)
    main(1024, 128)