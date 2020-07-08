"""This module handles loading the MAESTRO data set.

This module provides a collectio of functions for
loading and handling the MAESTRO data set.
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

import numpy as np
from audio_synthesis.utils.spectral import waveform_2_spectogram

# CLIP_NUMBER_STD is defined in the origonal implementation of SpecGAN
# the idea is that we normalize the spectograms and then
# clip any value that lies further than three
# standard deviations from the mean.
_CLIP_NUMBER_STD = 3.


def get_maestro_waveform_dataset(path):
    """Loads the MAESTRO dataset from a given path.

    Paramaters:
        path: The path to the .npz file containing
            the MAESTRO data set.
    """

    maestro = np.load(path)['arr_0']
    return maestro


def get_maestro_magnitude_phase_dataset(path, frame_length=512, frame_step=128,
                                        log_magnitude=True, instantaneous_frequency=True):
    """Loads the spectral representation of the MAESTRO dataset.

    Paramaters:
        path: The path to the .npz file containing
            the MAESTRO data set.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.
        log_magnitude: If true, the log of the magnitude is returned.
        instantaneous_frequency: If true, in the instantaneous frequency
            is returned instead of the phase.
    """

    maestro = get_maestro_waveform_dataset(path)

    process_spectogram = lambda x: waveform_2_spectogram(
        x,
        frame_length=frame_length,
        frame_step=frame_step,
        log_magnitude=log_magnitude,
        instantaneous_frequency=instantaneous_frequency
    )

    maestro = np.array(list(map(process_spectogram, maestro)))
    return maestro

def get_maestro_spectogram_normalizing_constants(path, frame_length=512,
                                                 frame_step=128,
                                                 log_magnitude=True,
                                                 instantaneous_frequency=True):
    """Computes the spectral normalizing constants for MAESTRO.

    Computes the mean and standard deviation for the spectogram
    representation of the MAESTRO dataset.

    Paramaters:
        path: The path to the .npz file containing
            the MAESTRO data set.
        frame_length (samples): Length of the FFT windows.
        frame_step (samples): The shift in time after each
            FFT window.
        log_magnitude: If true, the log of the magnitude is considered.
        instantaneous_frequency: If true, in the instantaneous frequency
            is considered instead of the phase.
    """

    raw_maestro = get_maestro_magnitude_phase_dataset(
        path,
        frame_length=frame_length,
        frame_step=frame_step,
        log_magnitude=log_magnitude,
        instantaneous_frequency=instantaneous_frequency
    )

    raw_maestro_magnitude = raw_maestro[:, :, :, 0]
    raw_maestro_phase = raw_maestro[:, :, :, 1]

    maestro_magnitude_mean = np.mean(raw_maestro_magnitude, axis=0)
    maestro_magnitude_std = np.std(raw_maestro_magnitude, axis=0)
    maestro_phase_mean = np.mean(raw_maestro_phase, axis=0)
    maestro_phase_std = np.std(raw_maestro_phase, axis=0)

    return maestro_magnitude_mean, maestro_magnitude_std, maestro_phase_mean, maestro_phase_std

def normalize(spectrum, mean, std):
    """Normalize a given magnitude or phase specturm

    Paramaters:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a single spectrum with no channel
            dimention (i.e., only two dimentions).
        mean: The mean of the data.
        std: The standard deviation of the data.
    """

    norm = (spectrum - mean) / std
    norm /= _CLIP_NUMBER_STD
    norm = np.clip(norm, -1., 1.)
    return norm

def un_normalize(spectrum, mean, std):
    """Un-normalize a given (normalized) magnitude or phase spectrum.

    Paramaters:
        spectrum: The magnitude spectrum to be un-normalized.
            It is expected to be a single spectrum with no channel
            dimention (i.e., only two dimentions).
        mean: The mean stastic that was used for normalizing
        std: The standard deviation stastic that was used for
            normalizing.
    """

    assert len(spectrum.shape) == 2
    spectrum = spectrum * _CLIP_NUMBER_STD
    spectrum = (spectrum * std) + mean
    return spectrum
