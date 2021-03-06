# Lint as: python3
"""Tests for spectral."""

import tensorflow as tf
import numpy as np
import os

import layers


class LayersTest(tf.test.TestCase):

    def test_conv_transpose_shape(self):
        inputs = np.random.normal(size=(10, 5, 2)).astype(np.float32)
        conv_transpose = layers.Conv1DTranspose(
            filters=2, kernel_size=1, strides=1
        )
        
        outputs = conv_transpose(inputs)
        self.assertShapeEqual(inputs, outputs)
        
    def test_conv_transpose_shape_upscale(self):
        inputs = np.random.normal(size=(10, 5, 2)).astype(np.float32)
        conv_transpose = layers.Conv1DTranspose(
            filters=2, kernel_size=1, strides=2
        )
        
        outputs = conv_transpose(inputs)
        self.assertEqual((10, 10, 2), outputs.shape)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
