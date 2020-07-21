# Lint as: python3
"""Tests for spectral."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import spectral


class SpectralTest(tf.test.TestCase):

    def test_waveform_to_spectogram_shape(self):
        waveform = np.random.normal(size=(2**14,))
        spectogram = spectral.waveform_2_spectogram(waveform, frame_length=512, frame_step=128)
        
        self.assertEqual((128, 256, 2), spectogram.shape)
        
    def test_waveform_to_magnitude_shape(self):
        waveform = np.random.normal(size=(2**14,))
        magnitude = spectral.waveform_2_magnitude(waveform, frame_length=512, frame_step=128)
        
        self.assertEqual((128, 256), magnitude.shape)
        
    def test_waveform_to_spectogram_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14))
        spectogram = spectral.waveform_2_spectogram(waveform, frame_length=512, frame_step=128)
        waveform_hat = spectral.spectogram_2_waveform(spectogram, frame_length=512, frame_step=128)
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        
    def test_waveform_to_magnitude_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14))
        spectogram = spectral.waveform_2_magnitude(waveform, frame_length=512, frame_step=128)
        waveform_hat = spectral.magnitude_2_waveform(spectogram, frame_length=512, frame_step=128)
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
