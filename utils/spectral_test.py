# Lint as: python3
"""Tests for spectral."""

import tensorflow as tf
import numpy as np
import os

import spectral


class SpectralTest(tf.test.TestCase):

    def test_waveform_to_stft_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_stft(waveform, frame_length=512, frame_step=128)[0]
        
        self.assertEqual((128, 256, 2), spectogram.shape)
        
    def test_waveform_to_mel_stft_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_stft(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]
        
        self.assertEqual((128, 80, 2), spectogram.shape)
    
    def test_waveform_to_spectogram_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(waveform, frame_length=512, frame_step=128)[0]
        
        self.assertEqual((128, 256, 2), spectogram.shape)
        
    def test_waveform_to_mel_spectogram_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]
        
        self.assertEqual((128, 80, 2), spectogram.shape)
        
    def test_waveform_to_magnitude_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        magnitude = spectral.waveform_2_magnitude(waveform, frame_length=512, frame_step=128)[0]
        
        self.assertEqual((128, 256), magnitude.shape)
        
    def test_waveform_to_mel_magnitude_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        magnitude = spectral.waveform_2_magnitude(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]
        
        self.assertEqual((128, 80), magnitude.shape)
        
    def test_waveform_to_spectogram_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(waveform, frame_length=512, frame_step=128)
        waveform_hat = spectral.spectogram_2_waveform(spectogram, frame_length=512, frame_step=128)[0]
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        
    def test_waveform_to_spectogram_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(waveform, frame_length=512, frame_step=128, n_mel_bins=80)
        waveform_hat = spectral.spectogram_2_waveform(spectogram, frame_length=512, frame_step=128, n_mel_bins=80)[0]
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        
    def test_waveform_to_stft_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14)).astype(np.float32)
        stft = spectral.waveform_2_stft(waveform, frame_length=512, frame_step=128)
        waveform_hat = spectral.stft_2_waveform(stft, frame_length=512, frame_step=128)[0]
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        
    def test_waveform_to_magnitude_return(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 2**14)).astype(np.float32)
        spectogram = spectral.waveform_2_magnitude(waveform, frame_length=512, frame_step=128)
        waveform_hat = spectral.magnitude_2_waveform(spectogram, frame_length=512, frame_step=128)[0]
        
        # Account for extra samples from reverse transform
        waveform_hat = waveform[0:len(waveform)]
        
        self.assertAllClose(waveform, waveform_hat)
        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
