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

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import signal as tf_signal
import scipy

def _get_window(frame_length):
    """Returns a window used for applying the filterbank and
    when reconstructing the signal.

    Args:
        frame_length: The length of the window in samples.

    Returns:
        window: The constructed window. The sum of the window
            squared is one.
    """

    window = scipy.signal.triang(frame_length)
    #window = window / np.sum(window)
    window = np.sqrt(window)
    return window

def _frame_signal(x_signal, frame_length, frame_step):
    """Takes a time-domain signal, separates it into frames,
    and applys a window.

    Args:

    """

    window = _get_window(frame_length)

    framed_signal = tf_signal.shape_ops.frame(x_signal, frame_length, frame_step)
    return framed_signal * window

def _overlap_and_add(framed_signal, frame_length, frame_step):
    """Takes a framed operation, and uses the overlap and add operation
    to reconstruct the signal origonal signal.


    """

    window = _get_window(frame_length)

    framed_signal = framed_signal * window
    return tf_signal.reconstruction_ops.overlap_and_add(framed_signal, frame_step)

def inner_op(signal_frames, filterbank):
    """
    """

    return tf.matmul(signal_frames, filterbank)

def apply_filterbank(x, frame_length, frame_step, filterbank):
    """
    """

    framed_signal = _frame_signal(x, frame_length, frame_step)
    return inner_op(framed_signal, filterbank)
    

def reconstruct(x_framed, frame_length, frame_step, filterbank_inv):
    """
    """

    framed_signal = inner_op(x_framed, filterbank_inv)
    return _overlap_and_add(framed_signal, frame_length, frame_step)

def erbs(f):
    """
    """

    return 21.4 * np.log10(1 + 0.00437 * f)

def inverse_erbs(erb_num):
    return (10**(erb_num / 21.4) - 1) / 0.00437

def erb(f):
    f_khz = f / 1000.0
    return 24.7 * (4.37 * f_khz + 1)

def band_pass_filter(center_freq, bandwidth, filter_length, sr=16000.0):
    bandwidth = bandwidth# + np.random.uniform(low=-0.05 * bandwidth, high=0.05 * bandwidth)
    bw = bandwidth / 2.0
    N = filter_length
    n = np.arange(-(N//2), N//2)

    k = np.sinc(2 * bw / sr * n + np.random.uniform(low=0, high=2*np.pi))

    w = np.hamming(N)
    h = k# * w

    s = np.sin(center_freq / sr * 2 * np.pi * n + np.random.uniform(low=0, high=2*np.pi))

    bp = h * s

    return bp

def get_filterbank(n_filters, filter_length, max_frequency):
    np.random.seed(1234)
    # Numbers refer to step in Basitaans algorythm
    max_erbs = erbs(max_frequency) #1
    erbs_frequencies = np.linspace(0, max_erbs, num=n_filters) #3
    #print(erbs_frequencies.shape)
    cfreqs = inverse_erbs(erbs_frequencies) #4
    #print(cfreqs.shape)
    bwidths = erb(cfreqs) #5
    #print(list(zip(cfreqs, bwidths)))

    filter_mat = np.zeros((len(cfreqs), filter_length))
    for i in range(0, len(cfreqs)): #7
        h_bp = band_pass_filter(cfreqs[i], bwidths[i], filter_length) #7.1 & #7.2
        filter_mat[i] = h_bp #10

    return filter_mat.astype(np.float32), bwidths, cfreqs


