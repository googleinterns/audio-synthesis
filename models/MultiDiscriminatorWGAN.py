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

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
import time
import os

def _compute_losses(model, d_real, d_fake, interpolated):        
    wasserstein_distance = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
            
    with tf.GradientTape() as t:
        t.watch(interpolated)
        d_interpolated = model.discriminator(interpolated, training=True)
            
    grad = t.gradient(d_interpolated, [interpolated])[0]
    sum_axes = [i for i in range(1, len(model.d_in_data_shape))]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=sum_axes))
    gp = tf.reduce_mean((slopes - 1.0) ** 2.0)
            
    G_loss = tf.reduce_mean(d_fake)
    D_loss = wasserstein_distance + 10.0 * gp
    
    return G_loss, D_loss
    

class WGAN:
    """Implements the training procedure for Wasserstein GAN [1] with Gradient Penalty [2].
    
    This class abstracts the training procedure for an arbatrary Wasserstein GAN [1] using
    the Gradient Penalty [2] technique for enforcing the required Lipschitz contraint. The
    current implementation uses a uniform prior distrubution, U(-1, 1).
    
    [1] Wasserstein GAN - https://arxiv.org/abs/1701.07875.
    [2] Improved Training of Wasserstein GANs - https://arxiv.org/abs/1704.00028.
    
    
    """
    
    def __init__(self, raw_dataset, data_shapes, d_in_data_shapes, generator, discriminator, z_dim, generator_optimizer, discriminator_optimizer, 
                generator_training_ratio=5, batch_size=64, epochs=1, checkpoint_dir=None, 
                 epochs_per_save = 10, fn_compute_loss=_compute_losses, fn_save_examples=None, get_discriminator_input_representations=None):
        """Initilizes the WGAN class.
        
        Paramaters:
            raw_dataset: A numpy array containing the training dataset.
            data_shape: The shape of the data points, should start with -1
                    to signify the batch size.
            d_in_data_shape: The shape of the data points at the input to
                    the discriminator, usually has an extra channel dimention.
            generator: The generator model.
            discriminator: The discriminator model.
            z_dim: The number of latent features.
            generator_optimizer: The optimizer for the generator model.
            discriminator_optimizer: The discriminator for the discriminator.
            generator_training_ratio: The number of discriminator updates 
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
        self.data_shapes = data_shapes
        self.d_in_data_shapes = d_in_data_shapes
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_training_ratio = generator_training_ratio
        self.batch_size = batch_size
        self.buffer_size = 1000
        self.epochs = epochs
        self.epochs_per_save = epochs_per_save
        self.fn_compute_loss = fn_compute_loss
        self.fn_save_examples = fn_save_examples
        self.get_discriminator_input_representations = get_discriminator_input_representations
        
        self.dataset = tf.data.Dataset.from_tensor_slices(self.raw_dataset).\
            shuffle(self.buffer_size).repeat(self.generator_training_ratio).batch(self.batch_size, drop_remainder=False)
        
        
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)
            
            
    def _train_step(self, X, train_generator=True, train_discriminator=True):
        """Executes one training step of the WGAN model.
        
        Paramaters:
            X: One batch of training data.
            train_generator: If true, the generator weights will be updated.
            train_discriminator: If true, the discriminator weights will be updated.
        """
        
        
        X_representations = self.get_discriminator_input_representations(X)
        
        Z = tf.random.uniform(shape=(X.shape[0], self.z_dim), minval=-1, maxval=1)

        with tf.GradientTape() as gen_tape:
            X_gen = self.generator(Z, training=True)
            X_gen_representations = self.get_discriminator_input_representations(X_gen)
            
            g_loss = 0
            d_loss = 0
            
            lambdas = [1.0, 1.0/1000.0]
            
            # Iterate through discriminators
            for i in range(len(self.discriminator)):
                with tf.GradientTape() as disc_tape:
                    # Compute Wasserstein Distance 
                    d_real = self.discriminator[i](X_representations[i], training=True)
                    d_fake = self.discriminator[i](X_gen_representations[i], training=True)
                
                    alpha_shape = np.ones(len(self.d_in_data_shapes[i]))
                    alpha_shape[0] = X_representations[i].shape[0]
                    alpha = tf.random.uniform(alpha_shape.astype('int32'), 0.0, 1.0)
                    diff = X_gen_representations[i] - X_representations[i]
                    interp = X_representations[i] + (alpha * diff)
                    
                    g_loss_i, d_loss_i = self.fn_compute_loss(self.discriminator[i], d_real, d_fake, interp)
                
                g_loss += lambdas[i] * g_loss_i
                gradients_of_discriminator = disc_tape.gradient(d_loss_i, self.discriminator[i].trainable_variables)
                
                if train_discriminator:
                    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator[i].trainable_variables))
            
            
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        if train_generator:
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        return g_loss, d_loss
            
    def _generate_and_save_examples(self, epoch, X_real):
        if self.fn_save_examples:
            z = tf.random.uniform(shape=(X_real.shape[0], self.z_dim), minval=-1, maxval=1)
            generations = self.generator(z, training=False)
            self.fn_save_examples(epoch, X_real, generations)
            
    def restore(self, checkpoint, completed_epochs):
        """Restores the model from a checkpoint

        Paramaters:
            checkpoint: The name of the checkpoint.
            completed_epochs: The number of training
                epochs completed by this checkpoint.
        """
        checkpoint_path = self.checkpoint_dir + checkpoint
        self.checkpoint.restore(checkpoint_path)
        self.completed_epochs = completed_epochs
        print("Checkpoint ", checkpoint_path,
              ' restored at ', str(self.completed_epochs), ' epochs')
        
    def train(self):
        """Executes the training for the WGAN model.
        """
        self._generate_and_save_examples(0, self.raw_dataset[0:self.batch_size])
        for epoch in range(self.epochs):
            pb_i = Progbar(len(self.raw_dataset))
            start = time.time()

            i = 1
            for X in self.dataset:
                G_loss, D_loss = self._train_step(X, train_generator=False, train_discriminator=True)
                if i % self.generator_training_ratio == 0:
                    g_loss, d_loss = self._train_step(X, train_generator=True, train_discriminator=False)
                    pb_i.add(self.batch_size)

                i += 1

            if self.checkpoint_prefix and (epoch + 1) % self.epochs_per_save == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print ('\nTime for epoch {} is {} minutes'.format(epoch + 1, (time.time()-start) / 60))
            self._generate_and_save_examples(epoch+1, X)
        
        
            
        