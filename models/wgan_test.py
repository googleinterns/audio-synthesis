# Lint as: python3
"""Tests for WGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import wgan


class SpectralTest(tf.test.TestCase):

    def test_interpolation_2d(self):
        x1 = np.random.normal(size=(10, 256))
        x2 = np.random.normal(size=(10, 256))
        
        interpolation = wgan._get_interpolation(x1, x2)
        self.assertShapeEqual(x1, interpolation)
        
    def test_interpolation_3d(self):
        x1 = np.random.normal(size=(10, 256, 32))
        x2 = np.random.normal(size=(10, 256, 32))
        
        interpolation = wgan._get_interpolation(x1, x2)
        self.assertShapeEqual(x1, interpolation)

        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
