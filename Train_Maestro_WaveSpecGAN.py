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
from structures.WaveGAN import Generator, Discriminator
from structures.SpecGAN import Discriminator as SpecDiscriminator
from datasets.MAESTRODataset import get_maestro_waveform_dataset
from models.MultiDiscriminatorWGAN import WGAN
import time
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 64
BATCH_SIZE = 64
EPOCHS = 300

# Setup Dataset
maestro_path = 'data/MAESTRO_6h.npz'
raw_maestro = get_maestro_waveform_dataset(maestro_path)

# Construct generator and discriminator
generator = Generator()
discriminator = Discriminator()
spec_discriminator = SpecDiscriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = '_results/representation_study/WaveSpecGAN/training_checkpoints/'

def _get_discriminator_input_representations(x):
    stft = tf.signal.stft(tf.reshape(x, (-1, 2**14)), frame_length=256, frame_step=128, pad_end=True)
    magnitude = tf.abs(stft)
    magnitude = tf.math.log(magnitude + 1e-6)
    magnitude = magnitude[:,:,0:-1]
    magnitude = tf.expand_dims(magnitude, axis=3)

    return (tf.reshape(x, (-1, 2**14, 1)), magnitude)

def _compute_losses(discriminator, d_real, d_fake, interpolated):
    wasserstein_distance_waveform = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
    
    # Compute waveform GP
    with tf.GradientTape() as t:
        t.watch(interpolated)
        d_interp = discriminator(interpolated, training=True)
            
    grad = t.gradient(d_interp, [interpolated])[0]
    redu_axis = list(range(len(interpolated.shape)))[1:]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=redu_axis))
    gp = tf.reduce_mean((slopes - 1.0) ** 2.0)
    
    # Compute and return losses
    g_loss = tf.reduce_mean(d_fake)
    d_loss = wasserstein_distance_waveform + 10.0 * gp
        
    return g_loss, d_loss

def save_examples(epoch, real, generated):  
    real_waveforms = np.reshape(real, (-1))
    gen_waveforms = np.reshape(generated, (-1))

    sf.write('_results/representation_study/WaveSpecGAN/audio/real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write('_results/representation_study/WaveSpecGAN/audio/gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

    
WaveGAN = WGAN(raw_maestro, [[-1, 2**14], [-1, 128, 128]], [[-1, 2**14, 1], [-1, 128, 128, 1]], generator, [discriminator, spec_discriminator], Z_dim, generator_optimizer, discriminator_optimizer, generator_training_ratio=D_updates_per_g, batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=checkpoint_dir, fn_compute_loss=_compute_losses, fn_save_examples=save_examples, get_discriminator_input_representations=_get_discriminator_input_representations)

WaveGAN.train()