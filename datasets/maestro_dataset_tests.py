# Lint as: python3
"""Tests for maestro dataset."""

import tensorflow as tf
import numpy as np
import os

import maestro_dataset


class MaestroDatasetTest(tf.test.TestCase):

    def test_normalize_shape(self):
        inputs = np.random.normal(size=(10, 128, 256)).astype(np.float32)
        std = np.ones((128, 256))
        mean = np.zeros((128, 256))
        
        normalized_input = maestro_dataset.normalize(inputs, mean, std)

        self.assertEqual(inputs.shape, normalized_input.shape)
        
    def test_un_normalize_shape(self):
        inputs = np.random.normal(size=(10, 128, 256)).astype(np.float32)
        std = np.ones((128, 256))
        mean = np.zeros((128, 256))
        
        normalized_input = maestro_dataset.un_normalize(inputs, mean, std)

        self.assertEqual(inputs.shape, normalized_input.shape)
        
    def test_un_normalize_spectogram_shape(self):
        inputs = np.random.normal(size=(10, 128, 256, 2)).astype(np.float32)
        std = np.ones((128, 256))
        mean = np.zeros((128, 256))
        
        normalized_input = maestro_dataset.un_normalize_spectogram(inputs, [mean, std], [mean, std])

        self.assertEqual(inputs.shape, normalized_input.shape)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
