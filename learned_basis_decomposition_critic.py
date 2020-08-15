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

"""Training script for the learned basis function decomposition.
The loss is the L2 error, with an additional critic based loss.
"""

import os
import copy
import tensorflow as tf
import numpy as np
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models import learned_basis_decomposition, wgan
from audio_synthesis.structures import learned_basis_function

FILTER_LENGTH = 32
NUMBER_OF_FILTERS = 512
BATCH_SIZE = 64
EPOCHS = 100
D_UPDATES_RATIO = 5

# Setup Dataset
MAESTRO_PATH = 'data/MAESTRO_6h.npz'
CHECKPOINT_DIR = '_results/learned_decomposition/L2_and_critic/training_checkpoints/'
RESULTS_DIR = '_results/learned_decomposition/L2_and_critic/audio/'

def compute_auxiliary_loss(model, num_steps, x_in, decomposition, x_hat,
                           auxiliary_models, auxiliary_optimizers):
    """Computes the auxiliary classification loss for the learned
    decomposition model. Follows the specification given in the file
    'models/learned_basis_decomposition.py'

    Args:
        model: The learned basis function model
        num_steps: The number of training steps past this epoch.
        x_in: The batch of training data, a tuple, (signal, midi)
        decomposition: The encoder decomposition of the signals
            computed through the encoder.
        x_hat: The reconstction of the input signals, through
            decoder(encoder(x_in[0])).
        auxiliary_models: The list of auxiliary models, [classifier]
        auxiliary_optimizers: The list of auxiliary model optimizers.

    Returns:
        prediction_error: The auxiliary prediction error loss.
        train_enc_dec: True every 'D_UPDATES_PER_G' training steps.
    """

    discriminator = auxiliary_models[0]
    disc_optimizer = auxiliary_optimizers[0]
    x_discriminator_in = x_in[1]
    with tf.GradientTape() as disc_tape:
        x_discriminator_in = tf.expand_dims(x_discriminator_in, 2)
        x_hat = tf.expand_dims(x_hat, 2)

        d_real = discriminator(x_discriminator_in)
        d_fake = discriminator(x_hat)

        interpolated = wgan.get_interpolation(x_discriminator_in, x_hat)

        g_loss, d_loss = wgan.compute_losses(
            discriminator, d_real, d_fake, interpolated
        )

    gradients_of_discriminator = disc_tape.gradient(
        d_loss, discriminator.trainable_variables
    )
    disc_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return g_loss, (num_steps % D_UPDATES_RATIO == 0)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)
    shuffled_raw_maestro = copy.copy(raw_maestro)
    np.random.shuffle(shuffled_raw_maestro)

    optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    encoder = learned_basis_function.Encoder(FILTER_LENGTH, NUMBER_OF_FILTERS)
    decoder = learned_basis_function.Decoder(FILTER_LENGTH)
    discriminator = learned_basis_function.Discriminator()

    learned_decomposition_model = learned_basis_decomposition.LearnedBasisDecomposition(
        encoder, decoder, optimizer, (raw_maestro, shuffled_raw_maestro), BATCH_SIZE, EPOCHS,
        CHECKPOINT_DIR, RESULTS_DIR, compute_auxiliary_loss_fn=compute_auxiliary_loss,
        auxiliary_models=[discriminator], auxiliary_optimizers=[disc_optimizer],
        auxiliary_update_ratio=D_UPDATES_RATIO
    )

    learned_decomposition_model.train()

if __name__ == '__main__':
    main()
