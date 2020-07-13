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

"""Implements the training process for a Wassersten GAN
with Gradient Penalty [https://arxiv.org/abs/1704.00028].
"""

import time
import os
import tensorflow as tf
from tensorflow.keras import utils as keras_utils
import numpy as np

SHUFFLE_BUFFER_SIZE = 1000

# A common choice for the gradient penalty weighting
# is 10.0
GRADIENT_PENALTY_LAMBDA = 10.0

def _compute_losses(discriminator, d_real, d_fake, interpolated):
    """Base implementation of the function that computes the WGAN
    generator and disciminator losses.

    Args:
        discriminator: The discriminator function.
        d_real: The discriminator score for the real data points.
        d_fake: The discriminator score for the fake data points.
        interpolated: The interpolation between the real and fake
            data points.

    Returns:
        g_loss: The loss for the generator function.
        d_loss: The loss for the discriminator function.
    """
    wasserstein_distance = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        d_interpolated = discriminator(interpolated, training=True)

    gradient = tape.gradient(d_interpolated, [interpolated])[0]
    sum_axes = list(range(1, len(interpolated.shape)))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=sum_axes))
    gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2.0)

    g_loss = tf.reduce_mean(d_fake)
    d_loss = wasserstein_distance + GRADIENT_PENALTY_LAMBDA * gradient_penalty

    return g_loss, d_loss

class WGAN: # pylint: disable=too-many-instance-attributes
    """Implements the training procedure for Wasserstein GAN [1] with Gradient Penalty [2].

    This class abstracts the training procedure for an arbatrary Wasserstein GAN [1] using
    the Gradient Penalty [2] technique for enforcing the required Lipschitz contraint. The
    current implementation uses a uniform prior distrubution, U(-1, 1).

    [1] Wasserstein GAN - https://arxiv.org/abs/1701.07875.
    [2] Improved Training of Wasserstein GANs - https://arxiv.org/abs/1704.00028.
    """

    def __init__(self, raw_dataset, d_in_data_shape, generator, # pylint: disable=too-many-arguments, too-many-locals
                 discriminator, z_dim, generator_optimizer, discriminator_optimizer,
                 discriminator_training_ratio=5, batch_size=64, epochs=1, checkpoint_dir=None,
                 epochs_per_save=10, fn_compute_loss=_compute_losses, fn_save_examples=None):
        """Initilizes the WGAN class.

        Paramaters:
            raw_dataset: A numpy array containing the training dataset.
            d_in_data_shape: The shape of the data points at the input to
                    the discriminator, usually has an extra channel dimention.
            generator: The generator model.
            discriminator: The discriminator model.
            z_dim: The number of latent features.
            generator_optimizer: The optimizer for the generator model.
            discriminator_optimizer: The discriminator for the discriminator.
            discriminator_training_ratio: The number of discriminator updates
                    per generator update. Default is 5.
            batch_size: Number of elements in each batch.
            epochs: Number of epochs of the training set.
            checkpoint_dir: Directory in which the model weights are saved. If
                    None, then the model is not saved.
            epochs_per_save: How often the model weights are saved.
            fn_compute_loss: The function that computes the generator and
                    discriminator loss. Must have signature
                    f(model, d_real, d_fake, interpolated).
            fn_save_examples: A function to save generations and real data,
                    called after every epoch.
        """
        self.raw_dataset = raw_dataset
        self.d_in_data_shape = d_in_data_shape
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_training_ratio = discriminator_training_ratio
        self.batch_size = batch_size
        self.buffer_size = SHUFFLE_BUFFER_SIZE
        self.epochs = epochs
        self.completed_epochs = 0
        self.epochs_per_save = epochs_per_save
        self.fn_compute_loss = fn_compute_loss
        self.fn_save_examples = fn_save_examples

        self.dataset = tf.data.Dataset.from_tensor_slices(self.raw_dataset).\
            shuffle(self.buffer_size).repeat(self.discriminator_training_ratio).\
            batch(self.batch_size, drop_remainder=False)


        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
            self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                generator=self.generator,
                discriminator=self.discriminator
            )


    def restore(self, checkpoint, completed_epochs):
        """Restores the model from a checkpoint at a given
        number of epochs.

        Paramaters:
            checkpoint: The name of the checkpoint.
            completed_epochs: The number of training
                epochs completed by this checkpoint.
        """
        checkpoint_path = self.checkpoint_dir + checkpoint
        self.checkpoint.restore(checkpoint_path)
        self.completed_epochs = completed_epochs
        print('Checkpoint ', checkpoint_path,
              ' restored at ', str(self.completed_epochs), ' epochs')

    def _train_step(self, x_in, train_generator=True, train_discriminator=True): # pylint: disable=too-many-locals
        """Executes one training step of the WGAN model.

        Paramaters:
            x_in: One batch of training data.
            train_generator: If true, the generator weights will be updated.
            train_discriminator: If true, the discriminator weights will be updated.

        Returns:
            g_loss: The generator loss
            d_loss: The discriminator loss
        """

        x_in = tf.reshape(x_in, shape=self.d_in_data_shape)
        z_in = tf.random.uniform((x_in.shape[0], self.z_dim), -1, 1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            x_gen = self.generator(z_in, training=True)
            x_gen = tf.reshape(x_gen, shape=self.d_in_data_shape)

            d_real = self.discriminator(x_in, training=True)
            d_fake = self.discriminator(x_gen, training=True)

            # Compute a linear interpolation of the real and generated
            # data, this is used to compute the gradient penalty.
            # https://arxiv.org/abs/1704.00028
            alpha_shape = np.ones(len(self.d_in_data_shape))
            alpha_shape[0] = x_in.shape[0]
            alpha = tf.random.uniform(alpha_shape.astype(tf.int32), 0.0, 1.0)
            diff = x_gen - x_in
            interp = x_in + (alpha * diff)

            g_loss, d_loss = self.fn_compute_loss(self.discriminator, d_real, d_fake, interp)

        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)

        if train_generator:
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
        if train_discriminator:
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return g_loss, d_loss

    def _generate_and_save_examples(self, epoch):
        if self.fn_save_examples:
            x_save = self.raw_dataset[np.random.randint(
                low=0, high=len(self.raw_dataset), size=self.batch_size
            )]
            z_in = tf.random.uniform((len(x_save), self.z_dim), -1, 1)
            generations = tf.squeeze(self.generator(z_in, training=False))
            self.fn_save_examples(epoch, x_save, generations)


    def train(self):
        """Executes the training for the WGAN model."""

        self._generate_and_save_examples(0)
        for epoch in range(self.completed_epochs, self.epochs):
            pb_i = keras_utils.Progbar(len(self.raw_dataset))
            start = time.time()

            for i, x_batch in enumerate(self.dataset):
                self._train_step(x_batch, train_generator=False,
                                 train_discriminator=True)
                if (i + 1) % self.discriminator_training_ratio == 0:
                    self._train_step(x_batch, train_generator=True,
                                     train_discriminator=False)
                    pb_i.add(self.batch_size)


            if self.checkpoint_prefix and (epoch + 1) % self.epochs_per_save == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('\nTime for epoch {} is {} minutes'.format(epoch + 1,
                                                             (time.time() - start) / 60))
            self._generate_and_save_examples(epoch + 1)
