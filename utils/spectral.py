"""Prodvides functionality for converting audio representations.
"""

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

def waveform_2_spectogram(waveform, frame_length=512,
                          frame_step=128, log_magnitude=True,
                          instantaneous_frequency=True):
    """Transforms a Waveform to a Spectogram.

    Returns the spectrogram for the given input. Note, this function
    clips the last frequency band to make the resulting dimension
    a power of two.

    Paramaters:
        waveform: the signal to be transformed. Expected
            shape is [time]
        frame_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.
        instantaneous_frequency: If true, the instantaneous frequency will,
            be returned instead of phase.

    Returns:
        A spectogram representation of the input waveform.
    """

    stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, pad_end=True)

    magnitude = tf.abs(stft)
    phase = tf.math.angle(stft)

    if log_magnitude:
        magnitude = tf.math.log(magnitude + 1e-6)

    if instantaneous_frequency:
        phase = np.unwrap(phase)
        phase = np.concatenate([np.expand_dims(phase[0, :], 0), np.diff(phase, axis=-2)], axis=-2)

    spectogram = np.concatenate([np.expand_dims(magnitude, 2),
                                 np.expand_dims(phase, 2)], axis=-1).astype('float32')

    # Cut of extra band. This makes it a power of 2, this is
    # also done in the papers
    return spectogram[:, 0:-1, :]

def waveform_2_magnitude(waveform, frame_length=512, frame_step=128, log_magnitude=True):
    """Transform a Waveform to a Magnitude Spectrum.

    This function is a wrapper for waveform_2_spectogram and removes
    the phase component.

    Paramaters:
        waveform: the signal to be transformed. Expected shape
            is [time].
        frame_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.

    Returns:
        A magnitude spectrum representation of the input waveform.
    """

    spectogram = waveform_2_spectogram(waveform, frame_length=frame_length,
                                       frame_step=frame_step,
                                       log_magnitude=log_magnitude)
    magnitude = spectogram[:, :, 0]
    return magnitude

def magnitude_2_waveform(magnitude, n_iter=16, frame_length=512,
                         frame_step=128, log_magnitude=True):
    """Transform a Magnitude Spectrum to a Waveform.

    Uses the Griffin-Lim algorythm, via the librosa implementation.

    Paramaters:
        magnitude: the magnitude spectrum to be transformed. Expected
            shape is [time, frequencies]
        n_iter: number of Griffin-Lim iterations to run.
        frame_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be assumed.

    Returns:
        A waveform representation of the input magnitude spectrum
        where the phase has been estimated using Griffin-Lim.
    """
    if log_magnitude:
        magnitude = np.exp(magnitude)

    # Add the removed band back in as zeros
    magnitude = np.pad(magnitude, [[0, 0], [0, 1]])

    return griffinlim(np.transpose(magnitude), n_iter=n_iter,
                      win_length=frame_length, hop_length=frame_step,
                      pad_mode='constant', center=False)


def spectogram_2_waveform(spectogram, frame_length=512, frame_step=128,
                          log_magnitude=True, instantaneous_frequency=True):
    """Transforms a Spectogram to a Waveform.

    Paramaters:
        spectogram: the spectogram to be transformed. Expected shape
            is [time, frequencies, 2]
        frame_length: The length of each fft.
        frame_step: Time increment after each fft, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, log-magnitude will be assumed.
        instantaneous_frequency: If true, it is assumed the input is
            instantaneous frequency and not phase.

    Returns:
        A waveform representation of the input spectogram.
    """

    magnitude = spectogram[:, :, 0]
    phase = spectogram[:, :, 1]

    if log_magnitude:
        magnitude = tf.math.exp(magnitude)

    if instantaneous_frequency:
        phase = tf.cumsum(phase, axis=-2)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

    # Add the removed band back in as zeros
    magnitude = tf.pad(magnitude, [[0, 0], [0, 1]], constant_values=0)
    phase = tf.pad(phase, [[0, 0], [0, 1]], constant_values=0)

    real = magnitude * tf.math.cos(phase)
    img = magnitude * tf.math.sin(phase)

    stft = tf.complex(real, img)

    waveform = tf.signal.inverse_stft(stft, frame_length=frame_length, frame_step=frame_step)
    return waveform
