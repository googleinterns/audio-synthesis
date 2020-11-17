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

"""Finds the scaling coefficients to convert the improved
representation frame into a 1-tight frame.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as keras_utils
from audio_synthesis.utils import improved_representation

NUM_FILTERS = 2048
FILTER_LENGTH = 256
N_UPDATES = 100000

def operator_norm(matrix):
    """Compute the operator of the input matrix.
    Note: This function coresponds to the operator norm,
    under specific chices, both ||x|| and ||Ax|| are 
    eqcledian norm.

    Args:
        matrix: The maxtrix to compute the operator
            norm of.

    Returns:
        The operator norm of matrix. In this case,
        it is the maximum singular value.
    """

    Sigma = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_max(Sigma)


def main():
    filters, _, _ = improved_representation.filterbank(NUM_FILTERS, FILTER_LENGTH, 8000.0)
    filters = filters / np.linalg.norm(filters, axis=0) # Normalizing helps with learning.

    outer_products = tf.constant(np.array([x.dot(x.T) for x in np.expand_dims(filters, 2)]).astype(np.float32))
    c_raw = tf.Variable(initial_value=np.random.normal(size=(NUM_FILTERS, 1, 1)).astype(np.float32), trainable=True)#np.log(initial_c), trainable=True)#

    c_optimizer = tf.keras.optimizers.Adam(0.0005)
    pb_i = keras_utils.Progbar(N_UPDATES)

    for i in range(N_UPDATES):
        with tf.GradientTape() as weight_tape:
            c = tf.math.softplus(c_raw) # Ensures they are positive
            frame_operator = tf.reduce_sum(tf.multiply(c, outer_products), axis=0)
            matrix_difference = tf.eye(FILTER_LENGTH) - frame_operator
            opn = operator_norm(matrix_difference)
        gradients_of_c = weight_tape.gradient(opn, [c_raw])
        c_optimizer.apply_gradients(zip(gradients_of_c, [c_raw]))

        pb_i.add(1)
        if i % 10 == 0:
            print(opn)
            print(tf.reduce_min(c_raw))
            print(tf.reduce_max(c_raw))
    
    raw_scaling_coefficients = np.array(tf.reshape(c_raw, (-1)))
    scaling_coefficients = np.log(1 + np.exp(raw_scaling_coefficients))
    print(scaling_coefficients.shape)
    print(scaling_coefficients)
    scaling_coefficients = np.expand_dims(scaling_coefficients, 1)

    print('Origonal Condition Number: ', np.linalg.cond(filters))
    tight_filters = np.sqrt(scaling_coefficients) * filters
    print('Scaled Condition Number: ', np.linalg.cond(tight_filters))

    np.savez('filters_and_scaling__square_03.npz', phi=filters, S=scaling_coefficients, Phi_scaled=tight_filters)


if __name__ == '__main__':
    main()