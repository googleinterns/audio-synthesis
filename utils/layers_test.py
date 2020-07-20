# Lint as: python3
"""Tests for spectral."""

import tensorflow as tf
import numpy as np

import layers


class LayersTest(tf.test.TestCase):

    def test_conv_transpose_shape(self):
        inputs = np.random.normal(size=(10, 5, 2))
        conv_transpose = layers.Conv1DTranspose(filters=2,
                                                kernel_size=1,
                                                strides=1)
        outputs = conv_transpose(inputs)
        self.assertShapeEqual(inputs, outputs)


if __name__ == '__main__':
    tf.test.main()