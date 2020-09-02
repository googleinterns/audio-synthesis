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

"""Tests for distortion helpers."""

import soundfile as sf
import tensorflow as tf
import numpy as np
import os

TEST_AUDIO_PATH = './data/speech.wav'
NOISY_TEST_AUDIO_PATH = './data/speech_bab_0dB.wav'
# Magic numbers, see https://github.com/ludlows/python-pesq
EXPECTED_PESQ = 1.08
MAXIMUM_PESQ = 4.64

import perceptual_helper

class PerceptualHelperTest(tf.test.TestCase):

    def test_pesq_metric(self):
        clean_audio, _ = sf.read(TEST_AUDIO_PATH)
        noisy_audio, _ = sf.read(NOISY_TEST_AUDIO_PATH)
        
        clean_clean_score = perceptual_helper.pesq_metric(
            clean_audio, clean_audio
        )
        self.assertEqual(np.round(clean_clean_score, 2), MAXIMUM_PESQ)
        
        clean_noisy_score = perceptual_helper.pesq_metric(
            clean_audio, noisy_audio
        )
        self.assertEqual(np.round(clean_noisy_score, 2), EXPECTED_PESQ)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
