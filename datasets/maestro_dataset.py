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

"""This module handles loading the MAESTRO data set.

This module provides a collection of functions for loading and handling the
MAESTRO data set.
"""

import numpy as np
from audio_synthesis.utils import spectral

# CLIP_NUMBER_STD is defined in the original implementation of SpecGAN
# the idea is that we normalize the spectrograms and then
# clip any value that lies further than three
# standard deviations from the mean.
_CLIP_NUMBER_STD = 3.
_PROCESSING_BATCH_SIZE = 100

def get_maestro_waveform_dataset(path):
    """Loads the MAESTRO dataset from a given path.

    Args:
        path: The path to the .npz file containing the MAESTRO data set.

    Returns:
        An array of waveform chunks loaded from the given path.
    """

    maestro = np.load(path)['arr_0']
    return maestro


def get_maestro_magnitude_phase_dataset(path, frame_length=512, frame_step=128,
                                        log_magnitude=True, instantaneous_frequency=True):
    """Loads the spectral representation of the MAESTRO dataset.

    Args:
        path: The path to the .npz file containing
            the MAESTRO data set.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.
        log_magnitude: If true, the log of the magnitude is returned.
        instantaneous_frequency: If true, in the instantaneous frequency
            is returned instead of the phase.

    Returns:
        The MAESTRO dataset as an array of spectograms.
    """

    maestro = get_maestro_waveform_dataset(path)

    process_spectogram = lambda x: spectral.waveform_2_spectogram(
        x,
        frame_length=frame_length,
        frame_step=frame_step,
        log_magnitude=log_magnitude,
        instantaneous_frequency=instantaneous_frequency
    )

    processed_maestro = np.array(process_spectogram(maestro[0:_PROCESSING_BATCH_SIZE]))
    for idx in range(_PROCESSING_BATCH_SIZE, len(maestro), _PROCESSING_BATCH_SIZE):
        datapoints = maestro[idx:idx+_PROCESSING_BATCH_SIZE]
        processed_maestro = np.concatenate([processed_maestro, process_spectogram(datapoints)], axis=0)

    magnitude_stats, phase_stats = _get_maestro_spectogram_normalizing_constants(processed_maestro)
    return processed_maestro, magnitude_stats, phase_stats

def get_maestro_stft_dataset(path, frame_length=512, frame_step=128):
    """Loads the STFT representation of the MAESTRO dataset.

    Args:
        path: The path to the .npz file containing
            the MAESTRO data set.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.

    Returns:
        The MAESTRO dataset as an array of spectograms.
    """

    maestro = get_maestro_waveform_dataset(path)

    process_spectogram = lambda x: spectral.waveform_2_stft(
        x,
        frame_length=frame_length,
        frame_step=frame_step
    )

    processed_maestro = np.array(process_spectogram(maestro[0:_PROCESSING_BATCH_SIZE]))
    for idx in range(_PROCESSING_BATCH_SIZE, len(maestro), _PROCESSING_BATCH_SIZE):
        datapoints = maestro[idx:idx+_PROCESSING_BATCH_SIZE]
        processed_maestro = np.concatenate([processed_maestro, process_spectogram(datapoints)], axis=0)

    return processed_maestro


def _get_maestro_spectogram_normalizing_constants(spectogram_data):
    """Computes the spectral normalizing constants for MAESTRO.

    An internal function, used when loadng the maestro dataset.
    Computes the mean and standard deviation for the spectogram
    representation of the MAESTRO dataset.

    Args:
        spectogram_data: The loaded MAESTRO dataset represented as
            a list of spectograms.

    Returns:
        Mean and standard deviation stastics for the MAESTRO spectogram
        dataset:
            [maestro_magnitude_mean,
            maestro_magnitude_std,
            maestro_phase_mean,
            maestro_phase_std]
    """

    raw_maestro_magnitude = spectogram_data[:, :, :, 0]
    raw_maestro_phase = spectogram_data[:, :, :, 1]

    maestro_magnitude_mean = np.mean(raw_maestro_magnitude, axis=0)
    maestro_magnitude_std = np.std(raw_maestro_magnitude, axis=0)
    maestro_phase_mean = np.mean(raw_maestro_phase, axis=0)
    maestro_phase_std = np.std(raw_maestro_phase, axis=0)

    return [maestro_magnitude_mean, maestro_magnitude_std], [maestro_phase_mean, maestro_phase_std]

def normalize(spectrum, mean, std):
    """Normalize a given magnitude or phase specturm by given stastics.

    If a value sits over three standard deviations from the mean,
    it is clipped. Hence, the output vales are between -1 and 1.

    Args:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a single spectrum with no channel
            dimention (i.e., only two dimentions) [time, frequency].
        mean: The mean of the data.
        std: The standard deviation of the data.

    Returns:
        A normalized phase or magnitude spectrum
    """

    norm = (spectrum - mean) / std
    norm /= _CLIP_NUMBER_STD
    norm = np.clip(norm, -1., 1.)
    return norm

def un_normalize(spectrum, mean, std):
    """Un-normalize a given (normalized) magnitude or phase spectrum.

    Args:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a single spectrum with no channel
            dimention (i.e., only two dimentions) [time, frequency].
        mean: The mean stastic that was used for normalizing
        std: The standard deviation stastic that was used for
            normalizing.

    Returns:
        An un-normalized magnitude or phase spectrum.
    """

    assert len(spectrum.shape) == 2
    spectrum = spectrum * _CLIP_NUMBER_STD
    spectrum = (spectrum * std) + mean
    return spectrum


def un_normalize_spectogram(spectogram, magnitude_stats, phase_stats):
    """Un-normalize a given spectogram acording to the given stastics

    Args:
        spectogram: The spectogram to be un-normalized
        magnitude_stats: The mean and standard deviation used to
            normalize the magnitude
        phase_stats: The mean and standard deviation used to
            normalize the phase

    Returns:
        An un-normalized spectogram of the same shape as the input.
    """

    magnitude = un_normalize(spectogram[:, :, 0], *magnitude_stats)
    phase = un_normalize(spectogram[:, :, 1], *phase_stats)

    return np.concatenate([np.expand_dims(magnitude, axis=2),
                           np.expand_dims(phase, axis=2)], axis=-1)
