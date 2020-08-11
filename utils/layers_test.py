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
     
    def test_pad_zeros_2d_shape(self):
        inputs = np.random.normal(size=(10, 5, 2, 1)).astype(np.float32)
        pad_zeros_layer = layers.PadZeros2D(num_zeros=(2,2))
        
        outputs = pad_zeros_layer(inputs)
        self.assertEqual((10, 10, 4, 1), outputs.shape)
        
    def test_pad_zeros_2d_shape_channels(self):
        inputs = np.random.normal(size=(10, 5, 2, 100)).astype(np.float32)
        pad_zeros_layer = layers.PadZeros2D(num_zeros=(2,2))
        
        outputs = pad_zeros_layer(inputs)
        self.assertEqual((10, 10, 4, 100), outputs.shape)
        
    def test_deformable_convolution_shape(self):
        inputs = np.random.normal(size=(3, 128, 256, 2)).astype(np.float32)
        deformable_convolution = layers.DeformableConvolution2D(16, (5, 5), 2)
        output = deformable_convolution(inputs)
        
        self.assertEqual((3, 128, 256, 16), output.shape)
        
    def test_get_pixel_values_at_point(self):
        N = 25
        array = np.arange(N)
        array = np.reshape(array, (1, 5, 5, 1))
        
        index = tf.reshape([0,1,2,3,4], (1, 1, 1, -1))
        gathered = layers._get_pixel_values_at_point(array, [index, index])
        gathered = np.reshape(gathered, (-1))
        self.assertAllEqual(gathered, [0, 6, 12, 18, 24])
        
    def test_harmonic_convolution_filter_shape(self):
        inputs = np.random.normal(size=(3, 128, 256, 2)).astype(np.float32)
        harmonic_convolution_filter


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
