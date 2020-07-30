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

def _compute_losses(discriminator, d_real, d_fake, interpolated_x, interpolated_c):
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

    with tf.GradientTape() as x_tape, tf.GradientTape() as c_tape:
        x_tape.watch(interpolated_x)
        c_tape.watch(interpolated_c)
        d_interpolated = discriminator(interpolated_x, interpolated_c, training=True)

    gradient = x_tape.gradient(d_interpolated, [interpolated_x])[0]
    sum_axes = list(range(1, len(interpolated_x.shape)))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=sum_axes))
    gradient_penalty_x = tf.reduce_mean((slopes - 1.0) ** 2.0)

    gradient = c_tape.gradient(d_interpolated, [interpolated_c])[0]
    sum_axes = list(range(1, len(interpolated_c.shape)))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=sum_axes))
    gradient_penalty_c = tf.reduce_mean((slopes - 1.0) ** 2.0)

    g_loss = tf.reduce_mean(d_fake)
    d_loss = wasserstein_distance + GRADIENT_PENALTY_LAMBDA * gradient_penalty_x +\
                GRADIENT_PENALTY_LAMBDA * gradient_penalty_c

    return g_loss, d_loss

def _get_representations(x_in):
    """The default function to get discriminator representations.
    Just an indentity function that returns the singleton array
    containing the input.
    """

    return [x_in]

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
                 discriminator_training_ratio=5, batch_size=64, epochs=1, lambdas=None, checkpoint_dir=None,
                 epochs_per_save=10, fn_compute_loss=_compute_losses,
                 fn_get_discriminator_input_representations=_get_representations,
                 fn_save_examples=None):
        """Initilizes the WGAN class.

        Paramaters:
            raw_dataset: A numpy array containing the training dataset.
            d_in_data_shape: The shape of the data points at the input to
                    the discriminator, usually has an extra channel dimention.
            generator: The generator model.
            discriminator: A list of discriminator models. If only one 
                discriminator then a singleton list should be given.
            z_dim: The number of latent features.
            generator_optimizer: The optimizer for the generator model.
            discriminator_optimizer: The discriminator for the discriminator.
            discriminator_training_ratio: The number of discriminator updates
                    per generator update. Default is 5.
            batch_size: Number of elements in each batch.
            epochs: Number of epochs of the training set.
            lambdas: The relative weightings for each discriminator
                component of the generator loss. Defaults to 1.0
            checkpoint_dir: Directory in which the model weights are saved. If
                    None, then the model is not saved.
            epochs_per_save: How often the model weights are saved.
            fn_compute_loss: The function that computes the generator and
                    discriminator loss. Must have signature
                    f(model, d_real, d_fake, interpolated).
            fn_get_discriminator_input_representations: A function that takes
                a data point (real and fake) and produces a list of representations,
                one for each discriminator. Default is an identity function.
                Signature expected is f(x_in), result should be an N element list
                of representations.
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
        self.lambdas = lambdas
        self.completed_epochs = 0
        self.epochs_per_save = epochs_per_save
        self.fn_compute_loss = fn_compute_loss
        self.fn_get_discriminator_input_representations = fn_get_discriminator_input_representations
        self.fn_save_examples = fn_save_examples

        self.dataset = tf.data.Dataset.from_tensor_slices(self.raw_dataset).shuffle(
            self.buffer_size).repeat(self.discriminator_training_ratio * 2).batch(
            self.batch_size * 2, drop_remainder=True)


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

    def _train_step(self, x_in, c_in, c_gen_in, train_generator=True, train_discriminator=True): # pylint: disable=too-many-locals
        """Executes one training step of the WGAN model.

        Paramaters:
            x_in: One batch of training data.
            train_generator: If true, the generator weights will be updated.
            train_discriminator: If true, the discriminator weights will be updated.
        """

        x_in_representations = self.fn_get_discriminator_input_representations(x_in)
        
        with tf.GradientTape() as gen_tape:
            g_loss = 0
            
            x_gen = self.generator(c_in, training=True)
            x_gen_representations = self.fn_get_discriminator_input_representations(x_gen)
            
            for i in range(len(self.discriminator)):
                with tf.GradientTape() as disc_tape:
                    x_gen_representation = tf.reshape(
                        x_gen_representations[i], shape=self.d_in_data_shape[i]
                    )
                    x_in_representation = tf.reshape(
                        x_in_representations[i], shape=self.d_in_data_shape[i]
                    )
                    
                    d_real = self.discriminator[i](x_in_representation, c_in, training=True)
                    d_fake = self.discriminator[i](x_gen_representation, c_gen_in, training=True)

                    # Compute a linear interpolation of the real and generated
                    # data, this is used to compute the gradient penalty.
                    # https://arxiv.org/abs/1704.00028
                    alpha_shape = np.ones(len(self.d_in_data_shape[i]))
                    alpha_shape[0] = x_in.shape[0]
                    alpha = tf.random.uniform(alpha_shape.astype(np.int32), 0.0, 1.0)
                    diff = x_gen_representation - x_in_representation
                    interpolation = x_in_representation + (alpha * diff)
                    
                    alpha_shape = np.ones(len(c_in.shape))
                    alpha_shape[0] = x_in.shape[0]
                    alpha = tf.random.uniform(alpha_shape.astype(np.int32), 0.0, 1.0)
                    diff = c_gen_in - c_in
                    interpolation_c = c_in + (alpha * diff)

                    g_loss_i, d_loss_i = self.fn_compute_loss(
                        self.discriminator[i], d_real, d_fake, interpolation, interpolation_c
                    )

                lambda_i = 1
                if self.lambdas:
                    lambda_i = self.lambdas[i]
                    
                g_loss += lambda_i * g_loss_i
                
                if train_discriminator:
                    gradients_of_discriminator = disc_tape.gradient(
                        d_loss_i, self.discriminator[i].trainable_variables
                    )
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator, self.discriminator[i].trainable_variables)
                    )
            
        if train_generator:
            gradients_of_generator = gen_tape.gradient(
                g_loss, self.generator.trainable_variables
            )
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables)
            )


    def _generate_and_save_examples(self, epoch):
        if self.fn_save_examples:
            x_save = self.raw_dataset[0][0:self.batch_size]
            c_save = self.raw_dataset[1][0:self.batch_size]
            
            generations = tf.squeeze(self.generator(c_save, training=False))
            self.fn_save_examples(epoch, x_save, generations)


    def train(self):
        """Executes the training for the WGAN model."""

        self._generate_and_save_examples(0)
        for epoch in range(self.completed_epochs, self.epochs):
            pb_i = keras_utils.Progbar(len(self.raw_dataset[0]))
            start = time.time()

            for i, batch in enumerate(self.dataset):
                x_batch, c_batch = batch
                c_gen_batch = c_batch[self.batch_size:]
                c_batch = c_batch[0:self.batch_size]
                x_batch = x_batch[0:self.batch_size]
                
                self._train_step(x_batch, c_batch, c_gen_batch, train_generator=False,
                                 train_discriminator=True)
                if (i + 1) % self.discriminator_training_ratio == 0:
                    self._train_step(x_batch, c_batch, c_gen_batch, train_generator=True,
                                     train_discriminator=False)
                    pb_i.add(self.batch_size)


            if self.checkpoint_prefix and (epoch + 1) % self.epochs_per_save == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('\nTime for epoch {} is {} minutes'.format(epoch + 1,
                                                             (time.time() - start) / 60))
            self._generate_and_save_examples(epoch + 1)
