# Lint as: python3
"""Tests for spectral."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

import spectral


class SpectralTest(tf.test.TestCase):

    def test_give_me_a_name(self):
        pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
