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

import tensorflow as tf
from audio_synthesis.models import wgan

SHUFFLE_BUFFER_SIZE = 300

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

    gradient_penalty_x = wgan.compute_gradient_penalty(
        lambda interpolated: discriminator(interpolated, interpolated_c),
        interpolated_x
    )
    
    gradient_penalty_c = wgan.compute_gradient_penalty(
        lambda interpolated: discriminator(interpolated_x, interpolated),
        interpolated_c
    )

    g_loss = tf.reduce_mean(d_fake)
    d_loss = wasserstein_distance + wgan.GRADIENT_PENALTY_LAMBDA * gradient_penalty_x +\
                wgan.GRADIENT_PENALTY_LAMBDA * gradient_penalty_c

    return g_loss, d_loss

class ConditionalWGAN(wgan.WGAN): # pylint: disable=too-many-instance-attributes
    """Implements the training procedure for Wasserstein GAN [1] with Gradient Penalty [2] in
    a conditional setting.

    This class extends the training procedure for an arbatrary Wasserstein GAN [1] using
    the Gradient Penalty [2] technique for enforcing the required Lipschitz contraint. The
    current implementation uses a uniform prior distrubution, U(-1, 1).

    [1] Wasserstein GAN - https://arxiv.org/abs/1701.07875.
    [2] Improved Training of Wasserstein GANs - https://arxiv.org/abs/1704.00028.
    """

    def __init__(self, raw_dataset, raw_conditioning_dataset, generator,
                 discriminator, z_dim, generator_optimizer, discriminator_optimizer,
                 discriminator_training_ratio=5, batch_size=64, epochs=1, checkpoint_dir=None,
                 epochs_per_save=10, fn_compute_loss=_compute_losses,
                 fn_get_discriminator_input_representations=wgan.get_representations,
                 fn_save_examples=None):
        """Initilizes the WGAN class.

        Paramaters:
            raw_dataset: A numpy array containing the training dataset.
            raw_conditioning_dataset: A numpy array containing the conditioning information.
                Should be aligned with raw_dataset, and contain the same number of
                elements.
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

        super(ConditionalWGAN, self).__init__(
            raw_conditioning_dataset, generator, discriminator, z_dim,
            generator_optimizer, discriminator_optimizer, discriminator_training_ratio,
            batch_size, epochs, checkpoint_dir, epochs_per_save, fn_compute_loss,
            fn_get_discriminator_input_representations, fn_save_examples
        )
        self.raw_x_dataset = raw_dataset
        
        self.conditioned_dataset = tf.data.Dataset.from_tensor_slices(
            (raw_dataset, raw_conditioning_dataset)).shuffle(
            self.buffer_size).repeat(self.discriminator_training_ratio).batch(
            self.batch_size, drop_remainder=False)

    def _train_step(self, data_in, train_generator=True, train_discriminator=True):
        """Executes one training step of the WGAN model.

        Paramaters:
            x_in: One batch of training data.
            train_generator: If true, the generator weights will be updated.
            train_discriminator: If true, the discriminator weights will be updated.
        """

        xc_in, c_gen_in = data_in
        x_in, c_in = xc_in
        
        x_in_representations = self.fn_get_discriminator_input_representations(x_in)

        with tf.GradientTape() as gen_tape:
            g_loss = 0

            z_in = tf.random.uniform((x_in.shape[0], self.z_dim), -1, 1)
            x_gen = self.generator(c_gen_in, z_in, training=True)
            x_gen_representations = self.fn_get_discriminator_input_representations(x_gen)

            for i in range(len(self.discriminator)):
                with tf.GradientTape() as disc_tape:
                    d_real = self.discriminator[i](
                        x_in_representations[i], c_in, training=True
                    )
                    d_fake = self.discriminator[i](
                        x_gen_representations[i], c_gen_in, training=True
                    )

                    x_interpolation = wgan.get_interpolation(
                        x_in_representations[i], x_gen_representations[i]
                    )
                    c_interpolation = wgan.get_interpolation(c_in, c_gen_in)

                    g_loss_i, d_loss_i = self.fn_compute_loss(
                        self.discriminator[i], d_real, d_fake, x_interpolation, c_interpolation
                    )

                g_loss += self.discriminator[i].weighting * g_loss_i

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

    def _get_training_dataset(self):
        """Function gives the dataset to use during training.
        In this case, returns the joint data/conditioning dataset
        with random conditioning information.

        Returns:
            A tf.Data dataset object for model training.
        """

        return zip(self.conditioned_dataset, self.dataset)

    def _generate_and_save_examples(self, epoch):
        if self.fn_save_examples:
            z_in = tf.random.uniform((self.batch_size, self.z_dim), -1, 1)
            x_save = self.raw_x_dataset[0:self.batch_size]
            c_save = self.raw_dataset[0:self.batch_size]

            generations = tf.squeeze(self.generator(c_save, z_in, training=False))
            self.fn_save_examples(epoch, x_save, generations)
