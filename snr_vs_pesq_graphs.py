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

"""Generates graphs of the PESQ for distorted representations
vs the SNR.
"""

import sys
import os
import errno
import tensorflow as tf
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pesq
import librosa
from tensorflow.keras import utils
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral

def mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def add_noise_at_snr_channel_with_average(channel_in, snr, n_avg):
    noisy_channel_in = add_noise_at_snr_channel(channel_in, snr)
    
    for i in range(n_avg-1):
        noisy_channel_in = noisy_channel_in + add_noise_at_snr_channel(channel_in, snr)
        
    return noisy_channel_in / n_avg
    
def add_noise_at_snr_channel(channel_in, snr):
    RMS = np.sqrt(np.mean(channel_in ** 2.0))
    RMS_noise = RMS / np.sqrt(10.0 ** (snr/10.0))
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

    #return np.expand_dims(np.concatenate([distorted_channels[0], distorted_channels[1]], axis=-1), 0)
    magnitude_only = np.concatenate([distorted_channels[0], orig_channels[1]], axis=-1)
    phase_only = np.concatenate([orig_channels[0], distorted_channels[1]], axis=-1)
    return magnitude_only, phase_only
    
def distort_one_channel_representation(representation, snr, n_avg=1):
    return np.expand_dims(add_noise_at_snr_channel_with_average(representation, snr, n_avg), 0)

def perceptual_error(reference, degraded): 
    reference = np.array(reference)
    degraded = np.array(degraded)
    
    degraded = degraded[0:len(reference)]
    return pesq.pesq(SAMPLE_RATE, reference, degraded, 'wb')

MAX_SNR = 45
MIN_SNR = 0
SNR_STEP = 2
SNRS = list(range(MIN_SNR, MAX_SNR+SNR_STEP, SNR_STEP))
SAMPLE_RATE = 16000
N_REPEATS = 20
LOG_MAGNITUDE = False
MEL_LOWER_HERTZ_EDGE = 0.0
MEL_UPPER_HERTZ_EDGE = 8000.0
WAVEFORM_PATH = './data/speech_8.wav'
RESULTS_PATH = './_results/snr_vs_pesq/'
LINE_STYLES = ['--', ':']

REPRESENTATIONS = {
    'STFT': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_stft(
            waveform, window_length, window_step
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.stft_2_waveform(
            representation, window_length, window_step
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    '(mel) STFT': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_stft(
            waveform, window_length, window_step, n_mel_bins*2, MEL_LOWER_HERTZ_EDGE, MEL_UPPER_HERTZ_EDGE
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.stft_2_waveform(
            representation, window_length, window_step, n_mel_bins*2, MEL_LOWER_HERTZ_EDGE, MEL_UPPER_HERTZ_EDGE
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Mag': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_magnitude(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.magnitude_2_waveform(
            representation, 32, window_length, window_step, log_magnitude=LOG_MAGNITUDE
        )[0],
        'distort_representation': distort_one_channel_representation,
    },
    '(mel) Mag': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_magnitude(
            waveform, window_length, window_step, LOG_MAGNITUDE, n_mel_bins*2, MEL_LOWER_HERTZ_EDGE, MEL_UPPER_HERTZ_EDGE
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.magnitude_2_waveform(
            representation, 32, window_length, window_step, LOG_MAGNITUDE, n_mel_bins*2, MEL_LOWER_HERTZ_EDGE, MEL_UPPER_HERTZ_EDGE
        )[0],
        'distort_representation': distort_one_channel_representation,
    },
    'Mag + Phase': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_spectogram(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=False
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.spectogram_2_waveform(
            representation, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=False
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Mag + IF': {
        'requires_fft_params': True,
        'waveform_2_representation': lambda waveform, window_length, window_step, n_mel_bins: spectral.waveform_2_spectogram(
            waveform, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=True
        )[0],
        'representation_2_waveform': lambda representation, window_length, window_step, n_mel_bins: spectral.spectogram_2_waveform(
            representation, window_length, window_step, log_magnitude=LOG_MAGNITUDE, instantaneous_frequency=True
        )[0],
        'distort_representation': distort_two_channel_representation,
    },
    'Waveform': {
        'requires_fft_params': False,
        'waveform_2_representation': lambda x: x,
        'representation_2_waveform': lambda x: x,
        'distort_representation': distort_one_channel_representation,
    },
    '(Avg) Waveform': {
        'requires_fft_params': False,
        'distort_with_fft_average': True,
        'waveform_2_representation': lambda x: x,
        'representation_2_waveform': lambda x: x,
        'distort_representation': distort_one_channel_representation,
    }
}

def process_representation_at_snr(representation, origonal_audio, snr, fft_window_size, fft_window_step):
    """Computes the PESQ score for the given representation after
    adding noise to achive a given SNR.
    
    Args:
        representation: The name of the representation being evaluated.
        origonal_audio: The audio being used for the evaluation.
        snr: The desired SNR.
        fft_window_size: The fft window size for spectral representations
        fft_window_step: The fft window step for spectral representations

    Returns:
        perceptual_errors: A list of PESQ scores asocitated with distorting each
            component of the representation to the given SNR.
        audio_hats: A list of time-domain waveforms asocitated with distorting each
            component of the representation to the given SNR.
    """
    
    if REPRESENTATIONS[representation]['requires_fft_params']:
        audio_representation = REPRESENTATIONS[representation]['waveform_2_representation'](
            origonal_audio, fft_window_size, fft_window_step, fft_window_size//2
        )
    else:
        audio_representation = REPRESENTATIONS[representation]['waveform_2_representation'](origonal_audio)
                
    if 'distort_with_fft_average' in REPRESENTATIONS[representation]:
        noisy_representations = REPRESENTATIONS[representation]['distort_representation'](
            audio_representation, snr, (fft_window_size // fft_window_step)
        )
    else:
        noisy_representations = REPRESENTATIONS[representation]['distort_representation'](
            audio_representation, snr
        )
            
    perceptual_errors = []
    audio_hats = []
    for noisy_representation in noisy_representations:
        if REPRESENTATIONS[representation]['requires_fft_params']:
            audio_hat = REPRESENTATIONS[representation]['representation_2_waveform'](
                noisy_representation, fft_window_size, fft_window_step, fft_window_size//2
            )[0:len(origonal_audio)]
        else:
            audio_hat = REPRESENTATIONS[representation]['representation_2_waveform'](
                noisy_representation
            )[0:len(origonal_audio)]
            
        audio_hats.append(audio_hat)
        perceptual_errors.append(perceptual_error(origonal_audio, audio_hat))
    return perceptual_errors, audio_hats

def main(fft_window_size, fft_window_step):
    """Generates the audio and PESQ vs SNR graphs for a given STFT
    setup.
    
    Args:
        fft_window_size: The FFT window size.
        fft_window_step: The FFT window step.
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print(fft_window_size, ' ', fft_window_step, ' ', (fft_window_size//2))
    
    origonal_audio, sr = sf.read(WAVEFORM_PATH)
    origonal_audio = origonal_audio.astype(np.float32)
    print("Sanity Check PESQ: ", perceptual_error(origonal_audio, origonal_audio))
    
    for representation in REPRESENTATIONS:
        REPRESENTATIONS[representation]['perceptual_errors'] = []
        REPRESENTATIONS[representation]['waveforms'] = []
    
    for snr in SNRS:
        pb_i = utils.Progbar(len(REPRESENTATIONS) * N_REPEATS)
        print("SNR: ", snr)
        for representation in REPRESENTATIONS:            
            all_perceptual_errors = []
            for n in range(N_REPEATS):
                perceptual_errors, audio_hats = process_representation_at_snr(
                    representation, origonal_audio, snr, fft_window_size, fft_window_step
                )
                all_perceptual_errors.append(perceptual_errors)
                pb_i.add(1)
                
            print(' ', representation, ' -> ', np.mean(all_perceptual_errors, 0))
            REPRESENTATIONS[representation]['perceptual_errors'].append(np.mean(all_perceptual_errors, 0))
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
    plt.savefig(os.path.join(RESULTS_PATH, file_name), bbox_inches='tight', dpi=920)
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