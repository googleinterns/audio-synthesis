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
import numpy as np
from librosa.core import griffinlim

def waveform_2_spectogram(x, fft_length=512, frame_step=128, log_magnitude=True, instantaneous_frequency=True):
    """Transforms a Waveform to a Spectogram.
    
    Returns the spectrogram for the given input. Note, this function
    clips the last frequency band to make the resulting dimension
    a power of two.
    
    Paramaters:
        x: the signal to be transformed.
        fft_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=fft_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.
        instantaneous_frequency: If true, the instantaneous frequency will,
            be returned instead of phase.
    """
    
    stft = tf.signal.stft(x, frame_length=fft_length, frame_step=frame_step, pad_end=True)
        
    magnitude = tf.abs(stft)
    phase = tf.math.angle(stft)

    if log_magnitude:
        magnitude = np.log(magnitude + 1e-8)
            
    if instantaneous_frequency:
        phase = np.unwrap(phase)
        phase = np.concatenate([np.expand_dims(phase[0,:], 0), np.diff(phase, axis=-2)], axis=-2)
    
    spectogram = np.concatenate([np.expand_dims(magnitude, 2), np.expand_dims(phase, 2)], axis=-1).astype('float32')
    
    # Cut of extra band. This makes it a power of 2, this is
    # also done in the papers
    return spectogram[:,0:-1,:]

def waveform_2_magnitude(x, fft_length=512, frame_step=128, log_magnitude=True):
    """Transform a Waveform to a Magnitude Spectrum.
    
    Paramaters:
        x: the signal to be transformed.
        fft_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=fft_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.
    """
    
    spectogram = waveform_2_spectogram(x, fft_length=fft_length, frame_step=frame_step, log_magnitude=log_magnitude)
    magnitude = spectogram[:,:,:,0]
    return magnitude

def magnitude_2_waveform(magnitude, n_iter=16, fft_length=512, frame_step=128, log_magnitude=True):
    """Transform a Magnitude Spectrum to a Waveform.
    
    Uses the Griffin-Lim algorythm, via the librosa implementation.
    
    Paramaters:
        magnitude: the magnitude spectrum to be transformed.
        n_iter: number of Griffin-Lim iterations to run.
        fft_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=fft_length - frame_step.
        log_magnitude: If true, the log-magnitude will be assumed.
    """
    if log_magnitude:
        magnitude = np.exp(magnitude)
    
    # Add the removed band back in as zeros
    magnitude = np.pad(magnitude, [[0,0], [0,1]])
    
    return griffinlim(np.transpose(magnitude), n_iter=n_iter, win_length=fft_length, hop_length=frame_step, pad_mode='constant', center=False)
    
    
def spectogram_2_waveform(spectogram, fft_length=512, frame_step=128, log_magnitude=True, instantaneous_frequency=True):
    """Transforms a Spectogram to a Waveform.
    
    Paramaters:
        spectogram: the spectogram to be transformed.
        fft_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=fft_length - frame_step.
        log_magnitude: If true, log-magnitude will be assumed.
        instantaneous_frequency: If true, it is assumed the input is
            instantaneous frequency and not phase
    """
    
    magnitude = spectogram[:,:,0]
    phase = spectogram[:,:,1]
    
    if log_magnitude:
        magnitude = np.exp(magnitude)
        
    if instantaneous_frequency:
        phase = np.array(phase).cumsum(axis=-2)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        
    # Add the removed band back in as zeros
    magnitude = np.pad(magnitude, [[0,0], [0,1]])
    phase = np.pad(phase, [[0,0], [0,1]])
    
    real = magnitude * np.cos(phase)
    img = magnitude * np.sin(phase)

    stft = real + 1j * img
    
    waveform = tf.signal.inverse_stft(stft, frame_length=fft_length, frame_step=frame_step)
    return waveform