# Lint as: python3

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

"""Tests for spectral."""

import os
import tensorflow as tf
import soundfile as sf
import numpy as np

import spectral
import perceptual_helper

TEST_AUDIO_PATH = './data/speech.wav'

class SpectralTest(tf.test.TestCase):

    def setUp(self):
        super(SpectralTest, self).setUp()

        audio, _ = sf.read(TEST_AUDIO_PATH)
        self.waveform = audio.astype(np.float32)

    def test_waveform_to_stft_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_stft(
            waveform, frame_length=512, frame_step=128
        )[0]

        self.assertEqual((128, 256, 2), spectogram.shape)

    def test_waveform_to_mel_stft_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_stft(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]

        self.assertEqual((128, 80, 2), spectogram.shape)

    def test_waveform_to_spectogram_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(
            waveform, frame_length=512, frame_step=128
        )[0]

        self.assertEqual((128, 256, 2), spectogram.shape)

    def test_waveform_to_mel_spectogram_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        spectogram = spectral.waveform_2_spectogram(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]

        self.assertEqual((128, 80, 2), spectogram.shape)

    def test_waveform_to_magnitude_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        magnitude = spectral.waveform_2_magnitude(
            waveform, frame_length=512, frame_step=128
        )[0]

        self.assertEqual((128, 256), magnitude.shape)

    def test_waveform_to_mel_magnitude_shape(self):
        waveform = np.random.normal(size=(2**14,)).astype(np.float32)
        magnitude = spectral.waveform_2_magnitude(
            waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]

        self.assertEqual((128, 80), magnitude.shape)

    def test_waveform_to_spectogram_return(self):
        spectogram = spectral.waveform_2_spectogram(
            self.waveform, frame_length=512, frame_step=128
        )
        waveform_hat = spectral.spectogram_2_waveform(
            spectogram, frame_length=512, frame_step=128
        )[0]

        # Account for extra samples from reverse transform
        waveform_hat = waveform_hat[0:len(self.waveform)]

        self.assertAllClose(self.waveform, waveform_hat, rtol=1e-3, atol=1e-3)
        
    def test_waveform_to_mel_spectogram_return(self):
        spectogram = spectral.waveform_2_spectogram(
            self.waveform, frame_length=512, frame_step=128, n_mel_bins=80
        )
        waveform_hat = spectral.spectogram_2_waveform(
            spectogram, frame_length=512, frame_step=128, n_mel_bins=80
        )[0]

        # Account for extra samples from reverse transform
        waveform_hat = waveform_hat[0:len(self.waveform)]

        # Higher tolerance as information is lost
        # from the transform to mel and back
        self.assertAllClose(self.waveform, waveform_hat, rtol=0.3, atol=0.3)

    def test_waveform_to_stft_return(self):
        stft = spectral.waveform_2_stft(
            self.waveform, frame_length=512, frame_step=128
        )
        waveform_hat = spectral.stft_2_waveform(
            stft, frame_length=512, frame_step=128
        )[0]

        # Account for extra samples from reverse transform
        waveform_hat = waveform_hat[0:len(self.waveform)]
        self.assertAllClose(self.waveform, waveform_hat, rtol=1e-3, atol=1e-3)

    def test_waveform_to_magnitude_return(self):
        spectogram = spectral.waveform_2_magnitude(
            self.waveform, frame_length=512, frame_step=128
        )
        waveform_hat = spectral.magnitude_2_waveform(
            spectogram, frame_length=512, frame_step=128
        )[0]

        # Account for extra samples from reverse transform
        waveform_hat = waveform_hat[0:len(self.waveform)]

        # Using pesq as difference can be quite large for a select number of points
        # since the griffin lim algorythm is used.
        pesq = perceptual_helper.pesq_metric(self.waveform, waveform_hat)
        self.assertTrue(pesq > 3.8)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
