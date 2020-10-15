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

"""Exports utilities helpful for dealing with distorting
numpy arrays with additive white Gaussian noise.
"""

import numpy as np

def distort_one_channel_representation(channel_in, snr, n_avg):
    """Handles distorting a single channel representation.

    Args:
        channel_in: The channel to be distorted.
        snr: The desired snr (dB) of noise to be added
        n_avg: Number of times to average.

    Returns:
        Sum of channel and noise. If n_avg > 1 then the result
        is a sum of n_avg indepdent distorted channels divided
        by n_avg. Shape is [1, channel_in.shape*]
    """

    distorted_channel = distort_channel(
        channel_in, snr, n_avg
    )

    return np.expand_dims(distorted_channel, 0)

def distort_channel(channel_in, snr, n_avg=1):
    """Function for distorting a channel. Allows averaging
    over multiple distortions, raising the effective SNR. 
    No averaging by default.

    Args:
        channel_in: The channel to be distorted.
        snr: The desired snr (dB) of noise to be added
        n_avg: Number of times to average.

    Returns:
        Sum of channel and noise. If n_avg > 1 then the result
        is a sum of n_avg indepdent distorted channels divided
        by n_avg. Shape is [channel_in.shape]
    """

    noisy_channel = add_noise_at_snr(channel_in, snr)

    for _ in range(n_avg - 1):
        noisy_channel = noisy_channel + add_noise_at_snr(channel_in, snr)

    avg_noisy_channel = noisy_channel / float(n_avg)
    return avg_noisy_channel

def add_noise_at_snr(channel_in, snr):
    """Distortes the given input by adding noise to achieve a given SNR

    Args:
        channel_in: The channel to be distorted.
        snr: The desired SNR.

    Returns:
        Sum of channel_in and white Gaussian noise. Shape is
        channel_in.shape
    """

    rms_channel = np.sqrt(np.mean(channel_in ** 2.0))
    noise_std = rms_channel / np.sqrt(10.0 ** (snr/10.0))

    return channel_in + np.random.normal(size=channel_in.shape, scale=noise_std)

def add_weighted_noise_at_snr(channel_in, snr, weighting):
    """Distortes the given input by adding noise to achieve a given SNR

    Args:
        channel_in: The channel to be distorted.
        snr: The desired SNR.
        weighting: 

    Returns:
        Sum of channel_in and white Gaussian noise. Shape is
        channel_in.shape
    """

    rms_channel = np.sqrt(np.mean(channel_in ** 2.0))
    noise_std = rms_channel / np.sqrt(10.0 ** (snr/10.0))

    return channel_in + np.random.normal(size=channel_in.shape, scale=noise_std) * weighting


def distort_multiple_channel_representation(representation, snr, n_avg=1):
    """Independently distort each channel of the representation.

    Args:
        representation: The representation to be distorted. The last
            dimention is taken as the channel. Expected shape is
            [-1, -1, channels].
        snr: The desired SNR (dB).
        n_avg: The number of times to average. Default is 1, i.e., no averaging.

    Returns:
        A list of representations, each with only one channel distorted. Shape
        is [representation.shape[-1], representation.shape*].
    """

    distorted_representations = []
    for channel in range(representation.shape[-1]):
        distorted_channel = distort_channel(
            representation[:, :, channel:channel + 1],
            snr, n_avg=n_avg
        )
        distorted_representation = np.concatenate([
            representation[:, :, 0:channel],
            distorted_channel,
            representation[:, :, channel + 1:]
        ], axis=-1)
        distorted_representations.append(distorted_representation)

    return np.array(distorted_representations)
