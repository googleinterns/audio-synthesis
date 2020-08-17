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

"""Exports the model responsible for training the learned basis function
setup based on the Conv TasNet model. Note that here we are not interested in
source separation, only in learning the decomposition.
"""

import time
import os
import soundfile as sf
import tensorflow as tf
import numpy as np
from tensorflow.keras import utils

_SAMPLE_RATE = 16000
_SHUFFLE_BUFFER_SIZE = 1000

def _compute_auxiliary_loss_fn(num_steps, x_in, decomposition, x_hat,
                               auxiliary_models, auxiliary_optimizers):
    """Example of an implementation of the compute auxiliary loss function.
    This implementation does nothing but describes the paramaters.

    Args:
        num_steps: The number of training steps past this epoch.
        x_in: The batch of input data. May be a tuple,
            if multiple input streams are given. Waveform data
            is the first.
        decomposition: The decomposition of x_in[0] through the
            encoder.
        x_hat: The reconstruction of the input signals, through
            decoder(encoder(x_in[0])).
        auxiliary_models: The list of auxiliary models.
        auxiliary_optimizers: The list of optimizers for the auxiliary
            models.

    Returns:
        auxiliary_loss: The auxiliary loss computed by the function.
            A scalar value.
        update_enc_dec: True if the encoder/decoder weights should be
            updated, otherwise false. Allows for an uneven update rule,
            e.g. WGAN.

    """

    return 0, True

class LearnedBasisDecomposition:
    """Implements the training procedure for the learned basis decomposition
    model based on the Conv-TasNet setup.
    """

    def __init__(self, encoder, decoder, optimizer, raw_dataset,
                 batch_size, epochs, checkpoint_dir, results_dir,
                 compute_auxiliary_loss_fn=_compute_auxiliary_loss_fn,
                 auxiliary_models=None, auxiliary_optimizers=None,
                 auxiliary_update_ratio=1):
        """Initilizes the LearnedBasisDecomposition class.

        Args:
            encoder: The encoder model
            decoder: The decoder model
            optimizer: The optimizer for both the encoder and decoder
            raw_dataset: The raw dataset, an array of datapoints. Could be
                an n-ple (waveform_data, ....) where successive arrays of data
                are auxiliary data.
            batch_size: The batch size for training
            epochs: The number of epochs to train for
            checkpoint_dir: The directory in which to save training progress
            results_dir: The directory in which to save results as training
                progresses.
            compute_auxiliary_loss_fn: Function that computes an auxiliary loss
                component. Should have the folowing signature: 
                def _compute_auxiliary_loss_fn(
                    num_steps, x_in, decomposition, x_hat,
                    auxiliary_models, auxiliary_optimizers
                    ).
            auxiliary_models: A list of models used by the auxiliary loss function
            auxiliary_optimizers: A list of optimizers for the auxiliary models.
            auxiliary_update_ratio: Number of times the auxiliary models
                are updated each update of the encoder/decoder.
        """
        
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.raw_dataset = raw_dataset
        self.batch_size = batch_size
        self.buffer_size = _SHUFFLE_BUFFER_SIZE
        self.epochs = epochs
        self.completed_epochs = 0
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.compute_auxiliary_loss_fn = compute_auxiliary_loss_fn
        self.auxiliary_models = auxiliary_models
        self.auxiliary_optimizers = auxiliary_optimizers
        self.auxiliary_update_ratio = auxiliary_update_ratio
        self.contains_auxiliary_data = isinstance(self.raw_dataset, tuple)
        
        if self.contains_auxiliary_data:
            self.dataset_length = len(self.raw_dataset[0])
        else:
            self.dataset_length = len(self.raw_dataset)

        self.dataset = tf.data.Dataset.from_tensor_slices(self.raw_dataset).shuffle(
                self.buffer_size).repeat(self.auxiliary_update_ratio).batch(
                self.batch_size, drop_remainder=False)

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder,
            decoder=self.decoder
        )

    def train_step(self, num_steps, x_in):
        """Executes one training step

        Args:
            num_steps: The number of ellapsed steps this training epoch.
            x_in: The batch of training data to train on. Shape
                is (batch_size, signal_length, 1), or [(batch_size,
                signal_length, 1), auxiliary_data_shapes].

        Returns:
            train_enc_dec: True if the encoder and decoder weights
                were updated.
        """

        if self.contains_auxiliary_data:
            x_signal_in = x_in[0]
        else:
            x_signal_in = x_in
            
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            x_signal_noisy = x_signal_in + tf.random.normal(shape=x_signal_in.shape, stddev=0.01)
            decomposition = self.encoder(x_signal_noisy)
            x_signal_hat = self.decoder(decomposition)

            x_signal_hat = tf.squeeze(x_signal_hat)
            x_signal_in = tf.squeeze(x_signal_in)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.abs(x_signal_in - x_signal_hat), axis=[1])
            )

            auxiliary_loss, train_enc_dec = self.compute_auxiliary_loss_fn(
                num_steps, x_in, decomposition, x_signal_hat,
                self.auxiliary_models, self.auxiliary_optimizers
            )

            loss = reconstruction_loss + auxiliary_loss

        if train_enc_dec:
            gradients_of_encoder = enc_tape.gradient(loss, self.encoder.trainable_variables)
            gradients_of_decoder = dec_tape.gradient(loss, self.decoder.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients_of_encoder, self.encoder.trainable_variables)
            )
            self.optimizer.apply_gradients(
                zip(gradients_of_decoder, self.decoder.trainable_variables)
            )

        return train_enc_dec, loss

    def save_audio(self, x_batch, epoch):
        """Saves a batch of real and reconstructed audio

        Args:
            x_batch: The batch of real data to be saved. Expected
                shape is (-1, signal_length).
            epoch: The current number of training epochs
                ellapsed.
        """

        if self.contains_auxiliary_data:
            x_batch = x_batch[0]

        decomp = self.encoder(x_batch)
        x_hat = self.decoder(decomp)

        sf.write(
            os.path.join(self.results_dir, 'orig_{}.wav'.format(epoch)),
            np.reshape(x_batch, (-1)), _SAMPLE_RATE
        )
        sf.write(
            os.path.join(self.results_dir, 'recon_{}.wav'.format(epoch)),
            np.reshape(x_hat, (-1)), _SAMPLE_RATE
        )

    def restore(self, checkpoint, completed_epochs):
        """Restores the model from a checkpoint at a given
        number of epochs.

        Args:
            checkpoint: The name of the checkpoint.
            completed_epochs: The number of training
                epochs completed by this checkpoint.
        """

        checkpoint_path = self.checkpoint_dir + checkpoint
        self.checkpoint.restore(checkpoint_path)
        self.completed_epochs = completed_epochs
        print('Checkpoint ', checkpoint_path,
              ' restored at ', str(self.completed_epochs), ' epochs')

    def train(self):
        """The main training loop for the model."""

        for epoch in range(self.completed_epochs, self.epochs):
            pb_i = utils.Progbar(self.dataset_length)
            start = time.time()

            for step, x_batch in enumerate(self.dataset):
                trained_enc_dec, loss = self.train_step(step, x_batch)
                if trained_enc_dec:
                    pb_i.add(self.batch_size, [('loss', loss)])

            self.save_audio(x_batch, epoch)

            if self.checkpoint_prefix and (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('\nTime for epoch {} is {} minutes'.format(epoch + 1, (time.time() - start) / 60))
