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

"""Provides functionality for converting audio representations."""

import tensorflow as tf
import numpy as np
from librosa.core import griffinlim

_EPSILON = 1e-6
_SAMPLE_RATE = 16000

def waveform_2_spectogram(waveform, frame_length=512, frame_step=128,
                          log_magnitude=True, instantaneous_frequency=True,
                          n_mel_bins=None, mel_lower_hertz_edge=None,
                          mel_upper_hertz_edge=None):
    """Transforms a Waveform to a Spectogram.

    Returns the spectrogram for the given input. Note, this function
    clips the last frequency band to make the resulting dimension
    a power of two.

    Paramaters:
        waveform: the signal to be transformed. Expected
            shape is [time] or [batch, time]
        frame_length: The length of each stft frame.
        frame_step: Time increment after each frame, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.
        instantaneous_frequency: If true, the instantaneous frequency will,
            be returned instead of phase.
        n_mel_bins: If specified, a spectogram in the mel scale will be
            returned.
        mel_lower_hertz_edge: The minimum frequency to be included in the
            mel-spectogram
        mel_upper_hertz_edge: The highest frequency to be included in the
            mel-spectogram

    Returns:
        A spectogram representation of the input waveform.
    """


    if len(waveform.shape) == 1:
        waveform = tf.expand_dims(waveform, 0)

    stft = tf.signal.stft(
        waveform, frame_length=frame_length, frame_step=frame_step, pad_end=True
    )
    
    # Cut off extra band. This makes it a power of 2, this is
    # also done in the papers
    magnitude = tf.abs(stft)[:, :, 0:-1]
    phase = tf.math.angle(stft)[:, :, 0:-1]

    if n_mel_bins:
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            n_mel_bins, magnitude.shape[-1], _SAMPLE_RATE, mel_lower_hertz_edge,
            mel_upper_hertz_edge
        )

        magnitude = tf.tensordot(magnitude, linear_to_mel_weight_matrix, 1)
        phase = tf.tensordot(phase, linear_to_mel_weight_matrix, 1)

    if log_magnitude:
        magnitude = tf.math.log(magnitude + _EPSILON)

    if instantaneous_frequency:
        phase = np.unwrap(phase)
        phase = np.concatenate([np.expand_dims(phase[:, 0, :], axis=-2),
                                np.diff(phase, axis=-2)], axis=-2).astype(np.float32)

    spectogram = tf.concat([tf.expand_dims(magnitude, 3),
                            tf.expand_dims(phase, 3)], axis=-1)

    return spectogram

def waveform_2_magnitude(waveform, frame_length=512, frame_step=128, log_magnitude=True,
                         n_mel_bins=None, mel_lower_hertz_edge=None,
                         mel_upper_hertz_edge=None):
    """Transform a Waveform to a Magnitude Spectrum.

    This function is a wrapper for waveform_2_spectogram and removes
    the phase component.

    Paramaters:
        waveform: the signal to be transformed. Expected shape
            is [time], or [batch, time].
        frame_length: The length of each frame.
        frame_step: Time increment after each frame, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be returned.
        n_mel_bins: If specified, a magnitude spectrum in the mel scale
            will be returned.
        mel_lower_hertz_edge: The minimum frequency to be included in the
            mel-spectogram
        mel_upper_hertz_edge: The highest frequency to be included in the
            mel-spectogram

    Returns:
        A magnitude spectrum representation of the input waveform.
    """

    spectogram = waveform_2_spectogram(
        waveform, frame_length=frame_length, frame_step=frame_step,
        log_magnitude=log_magnitude, n_mel_bins=n_mel_bins,
        mel_lower_hertz_edge=mel_lower_hertz_edge,
        mel_upper_hertz_edge=mel_upper_hertz_edge
    )

    magnitude = spectogram[:, :, :, 0]
    return magnitude

def magnitude_2_waveform(magnitude, n_iter=16, frame_length=512,
                         frame_step=128, log_magnitude=True):
    """Transform a Magnitude Spectrum to a Waveform.

    Uses the Griffin-Lim algorythm, via the librosa implementation.

    Paramaters:
        magnitude: the magnitude spectrum to be transformed. Expected
            shape is [time, frequencies] or [batch, time, frequencies]
        n_iter: number of Griffin-Lim iterations to run.
        frame_length: The length of each frame.
        frame_step: Time increment after each frame, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, the log-magnitude will be assumed.

    Returns:
        A waveform representation of the input magnitude spectrum
        where the phase has been estimated using Griffin-Lim.
    """

    if len(magnitude.shape) == 2:
        magnitude = tf.expand_dims(magnitude, 0)

    if log_magnitude:
        magnitude = np.exp(magnitude) - _EPSILON

    # Add the removed band back in as zeros
    magnitude = np.pad(magnitude, [[0, 0], [0, 0], [0, 1]])

    to_waveform = lambda m: griffinlim(
        np.transpose(m), n_iter=n_iter, win_length=frame_length,
        hop_length=frame_step, pad_mode='constant', center=False
    )
    
    return np.array(list(map(to_waveform, magnitude)))

def spectogram_2_waveform(spectogram, frame_length=512, frame_step=128,
                          log_magnitude=True, instantaneous_frequency=True):
    """Transforms a Spectogram to a Waveform.

    Paramaters:
        spectogram: the spectogram to be transformed. Expected shape
            is [time, frequencies, 2] or [batch, time, frequencies, 2]
        frame_length: The length of each frame.
        frame_step: Time increment after each frame, i.e.
            overlap=frame_length - frame_step.
        log_magnitude: If true, log-magnitude will be assumed.
        instantaneous_frequency: If true, it is assumed the input is
            instantaneous frequency and not phase.

    Returns:
        A waveform representation of the input spectogram.
    """

    if len(spectogram.shape) == 3:
        spectogram = tf.expand_dims(spectogram, 0)

    magnitude = spectogram[:, :, :, 0]
    phase = spectogram[:, :, :, 1]

    if log_magnitude:
        magnitude = tf.math.exp(magnitude) - _EPSILON

    if instantaneous_frequency:
        phase = tf.cumsum(phase, axis=-2)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

    # Add the removed band back in as zeros
    magnitude = tf.pad(magnitude, [[0, 0], [0, 0], [0, 1]], constant_values=0)
    phase = tf.pad(phase, [[0, 0], [0, 0], [0, 1]], constant_values=0)

    real = magnitude * tf.math.cos(phase)
    img = magnitude * tf.math.sin(phase)

    stft = tf.complex(real, img)

    waveform = tf.signal.inverse_stft(stft, frame_length=frame_length, frame_step=frame_step)
    return waveform
