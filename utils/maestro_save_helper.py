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

"""This module exports functionality that helps with saving generated
waveforms.
"""

import os
import soundfile as sf
import numpy as np
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral

def save_wav_data(epoch, real, generated, sampling_rate, result_dir, get_waveform):
    """Saves a batch of real and generated data.

    Args:
        epoch: The number of training epochs when the generated
            data was produced.
        real: A batch of real data, waveform or time-frequency.
        generated: A batch of generated data, waveform or time-frequency.
        sampling_rate: The sampling rate (samples/second) of the audio.
        result_dir: The directory in which to save the real and generated
            audio.
        get_waveform: A function that transforms the given audio representation
            into a waveform.
    """

    gen_waveforms = []
    real_waveforms = []
    for real_representation, generated_representation in zip(real, generated):
        real_waveforms.extend(get_waveform(real_representation))
        gen_waveforms.extend(get_waveform(generated_representation))

    sf.write(os.path.join(result_dir, 'real_{}.wav'.format(epoch)), real_waveforms, sampling_rate)
    sf.write(os.path.join(result_dir, 'gen_{}.wav'.format(epoch)), gen_waveforms, sampling_rate)

def get_waveform_from_normalized_magnitude(magnitude, statistics, griffin_lim_iterations,
                                          frame_length, frame_step, log_magnitude=True):
    """Converts a normalized magnitude spectrum into a waveform.

    A wrapper for the 'magnitude_2_waveform' function
    that handles un-normalization.

    Args:
        magnitude: The normalized magnitude to be converted to
            a waveform. A single magnitude with no channel dimention.
        statistics: The stastics used during normalization. Expected form
            is [mean, standard deviation].
        griffin_lim_iterations: The number of Griffin-Lim iterations.
        frame_length: The FFT frame length.
        frame_step: The FFT frame step.
        log_magnitude: If the log of the magnitude was taken.

    Returns:
        Waveform representation of the normalized magnitude
        spectrum. The phase is estimated using griffin-lim.
    """

    magnitude = maestro_dataset.un_normalize(magnitude, *statistics)

    return spectral.magnitude_2_waveform(
        magnitude, griffin_lim_iterations, frame_length, frame_step,
        log_magnitude
    )

def get_waveform_from_normalized_spectogram(spectogram, statistics, frame_length,
                                            frame_step, log_magnitude=True,
                                            instantaneous_frequency=True):
    """Converts a normalized spectogram into a waveform.

    Wrapper for spectogram_2_waveform that handles un-normalization.

    Args:
        spectogram: The normalized spectogram to be converted into
            a waveform.
        statistics: The normalizaton stastics. Expected form is
            [[mean, standard deviation], [mean, standard deviation]],
            where the first set corresponds to the magnitude and the
            second corresponds to the phase.
        frame_length: The FFT frame length.
        frame_step: The FFT frame step.
        log_magnitude: If the log of the magnitude was taken.
        instantaneous_frequency: If the instantaneous frequency is used
            instead of the phase.

    Returns:
        Wavefrom representation of the input spectogram.

    """

    magnitude_stats, phase_stats = statistics

    magnitude = spectogram[:, :, 0]
    magnitude = maestro_dataset.un_normalize(magnitude, *magnitude_stats)

    phase = spectogram[:, :, 1]
    phase = maestro_dataset.un_normalize(phase, *phase_stats)

    un_normalized_spectogram = np.concatenate([
        np.expand_dims(magnitude, axis=2),
        np.expand_dims(phase, axis=2)], axis=-1)

    return spectral.spectogram_2_waveform(
        un_normalized_spectogram, frame_length, frame_step, log_magnitude,
        instantaneous_frequency
    )
