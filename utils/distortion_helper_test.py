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

import tensorflow as tf
import numpy as np
import os

import distortion_helper

class DistortionHelperTest(tf.test.TestCase):

    def test_add_noise_at_snr_shape(self):
        desired_snr = 0.0
        representation = np.random.uniform(size=(128, 256, 1)).astype(np.float32)
        
        distorted = distortion_helper.add_noise_at_snr(representation, desired_snr)
        noise = distorted - representation
        
        power_channel = np.mean(representation ** 2.0)
        power_noise = np.mean(noise ** 2.0)
        
        snr_actual = np.log10(power_channel / power_noise)
        
        self.assertEqual(desired_snr, np.round(snr_actual, 1))
                
    def test_add_noise_at_snr_db(self):
        representation = np.random.normal(size=(128, 256, 1)).astype(np.float32)
        distorted = distortion_helper.add_noise_at_snr(representation, 0)
        
        self.assertEqual(representation.shape, distorted.shape)
        
    def test_distort_one_channel_representation_shape(self):
        representation = np.random.normal(size=(128, 256, 1)).astype(np.float32)
        distorted = distortion_helper.distort_one_channel_representation(
            representation, 0, 2
        )
        
        self.assertEqual((1,) + representation.shape, distorted.shape)
        
    def test_distort_multiple_channel_representation_shape(self):
        representation = np.random.normal(size=(128, 256, 3)).astype(np.float32)
        distorted = distortion_helper.distort_multiple_channel_representation(
            representation, 0, 2
        )
        
        self.assertEqual((representation.shape[-1],) + representation.shape, distorted.shape)
        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
