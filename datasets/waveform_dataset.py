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

"""This module handles loading the a waveform data set.

This module provides a collection of functions for loading and handling a
waveform data set.
"""

import numpy as np
from audio_synthesis.utils import spectral

# CLIP_NUMBER_STD is defined in the original implementation of SpecGAN
# the idea is that we normalize the spectrograms and then
# clip any value that lies further than three
# standard deviations from the mean.
_CLIP_NUMBER_STD = 3.
_PROCESSING_BATCH_SIZE = 100
_EPSILON = 1e-6

def get_waveform_dataset(path):
    """Loads the waveform dataset from a given path.

    Args:
        path: The path to the .npz file containing the waveform data set.

    Returns:
        An array of waveform chunks loaded from the given path.
    """

    dataset = np.load(path)['arr_0']
    return dataset

def _get_pre_processed_dataset(path, pre_process_fn):
    """Handles efficiently pre-processing the dataset.
    Args:
        path: The path to the .npz file containing the dataset.
        pre_process_fn: Implements the pre-processing functionality.
            Expected signature is f(batch) -> processed_batch, where
            batch has shape (batch, waveform_length).
    Returns:
        The pre-processed dataset.
    """

    dataset = get_waveform_dataset(path)

    processed_dataset = np.array(pre_process_fn(dataset[0:_PROCESSING_BATCH_SIZE]))
    for idx in range(_PROCESSING_BATCH_SIZE, len(dataset), _PROCESSING_BATCH_SIZE):
        datapoints = dataset[idx:idx + _PROCESSING_BATCH_SIZE]
        processed_dataset = np.concatenate([processed_dataset, pre_process_fn(datapoints)], axis=0)

    return processed_dataset

def get_magnitude_phase_dataset(path, frame_length=512, frame_step=128,
                                log_magnitude=True, instantaneous_frequency=True):
    """Loads the spectral representation of the dataset.
    Args:
        path: The path to the .npz file containing
            the dataset.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.
        log_magnitude: If true, the log of the magnitude is returned.
        instantaneous_frequency: If true, in the instantaneous frequency
            is returned instead of the phase.
    Returns:
        The dataset as an array of spectograms.
    """

    process_spectogram = lambda x: spectral.waveform_2_spectogram(
        x,
        frame_length=frame_length,
        frame_step=frame_step,
        log_magnitude=log_magnitude,
        instantaneous_frequency=instantaneous_frequency
    )

    processed_dataset = _get_pre_processed_dataset(path, process_spectogram)

    magnitude_stats, phase_stats = _get_spectogram_normalizing_constants(processed_dataset)
    return processed_dataset, magnitude_stats, phase_stats

def get_stft_dataset(path, frame_length=512, frame_step=128):
    """Loads the STFT representation of the dataset.
    Args:
        path: The path to the .npz file containing
            the dataset.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.
    Returns:
        The dataset as an array of spectograms.
    """

    process_stft = lambda x: spectral.waveform_2_stft(
        x,
        frame_length=frame_length,
        frame_step=frame_step
    )

    processed_dataset = _get_pre_processed_dataset(path, process_stft)

    return processed_dataset

def _get_spectogram_normalizing_constants(spectogram_data):
    """Computes the spectral normalizing constants for a waveform dataset.

    An internal function, used when loadng the dataset.
    Computes the mean and standard deviation for the spectogram
    representation of the dataset.

    Args:
        spectogram_data: The loaded dataset represented as
            a list of spectograms.

    Returns:
        Mean and standard deviation stastics for the spectogram
        dataset:
            [magnitude_mean,
            magnitude_std,
            phase_mean,
            phase_std]
    """

    raw_magnitude = spectogram_data[:, :, :, 0]
    raw_phase = spectogram_data[:, :, :, 1]

    magnitude_mean = np.mean(raw_magnitude, axis=0)
    magnitude_std = np.std(raw_magnitude, axis=0)
    phase_mean = np.mean(raw_phase, axis=0)
    phase_std = np.std(raw_phase, axis=0)

    return [magnitude_mean, magnitude_std], [phase_mean, phase_std]

def normalize(spectrum, mean, std):
    """Normalize a given magnitude or phase specturm by given stastics.

    If a value sits over three standard deviations from the mean,
    it is clipped. Hence, the output vales are between -1 and 1.

    Args:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a spectrum with no channel
            dimention, [time, frequency] or [-1, time, frequency].
        mean: The mean of the data.
        std: The standard deviation of the data.

    Returns:
        A normalized phase or magnitude spectrum. Shape is
        [-1, time, frequency]
    """

    norm = (spectrum - mean) / (std + _EPSILON)
    norm /= _CLIP_NUMBER_STD
    norm = np.clip(norm, -1., 1.)
    return norm

def un_normalize(spectrum, mean, std):
    """Un-normalize a given (normalized) magnitude or phase spectrum.

    Args:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a single spectrum with no channel
            dimention [time, frequency] or [batch_size, time, frequency].
        mean: The mean stastic that was used for normalizing
        std: The standard deviation stastic that was used for
            normalizing.

    Returns:
        An un-normalized magnitude or phase spectrum. Shape is
        [-1, time, frequency]
    """

    if len(spectrum.shape) == 2:
        spectrum = np.expand_dims(spectrum, 0)
        
    assert len(spectrum.shape) == 3
    spectrum = spectrum * _CLIP_NUMBER_STD
    spectrum = (spectrum * std) + mean
    return spectrum


def un_normalize_spectogram(spectogram, magnitude_stats, phase_stats):
    """Un-normalize a given spectogram acording to the given stastics

    Args:
        spectogram: The spectogram (can be batched) to be un-normalized.
            Expected shape is [time, frequency, 2] 
            or [-1, time, frequency, 2]
        magnitude_stats: The mean and standard deviation used to
            normalize the magnitude
        phase_stats: The mean and standard deviation used to
            normalize the phase

    Returns:
        An un-normalized spectogram. Shape is
        [-1, time, frequency, 2].
    """

    if len(spectogram.shape) == 3:
        spectogram = np.expand_dims(spectogram, 0)
    
    magnitude = un_normalize(spectogram[:, :, :, 0], *magnitude_stats)
    phase = un_normalize(spectogram[:, :, :, 1], *phase_stats)

    return np.concatenate([np.expand_dims(magnitude, axis=3),
                           np.expand_dims(phase, axis=3)], axis=-1)
