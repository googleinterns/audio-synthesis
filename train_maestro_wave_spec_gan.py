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

import os
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.structures import wave_gan
from audio_synthesis.structures import spec_gan
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models.MultiDiscriminatorWGAN import WGAN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
MEL_SPECTOGRAM = False
MEL_LOWER_HERTZ_EDGE = 80.
MEL_UPPER_HERTZ_EDGE = 7600.
NUM_MEL_BINS = 96
CHECKPOINT_DIR = '_results/representation_study/WaveSpecGAN_HR/training_checkpoints/'
RESULT_DIR = '_results/representation_study/WaveSpecGAN_HR/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

def _get_discriminator_input_representations(x):
    stft = tf.signal.stft(tf.reshape(x, (-1, 2**14)), frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP, pad_end=True)
    magnitude = tf.abs(stft)
    
    if MEL_SPECTOGRAM:
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            NUM_MEL_BINS, magnitude.shape[-1], SAMPLE_RATE, MEL_LOWER_HERTZ_EDGE,
            MEL_UPPER_HERTZ_EDGE
        )
    
        magnitude = tf.tensordot(magnitude, linear_to_mel_weight_matrix, 1)
    
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

    sf.write(RESULT_DIR + 'real_' + str(epoch) + '.wav', real_waveforms, 16000)
    sf.write(RESULT_DIR + 'gen_' + str(epoch) + '.wav', gen_waveforms, 16000)

def main():
    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)

    generator = wave_gan.Generator()
    discriminator = wave_gan.Discriminator()
    spec_discriminator = spec_gan.Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    
    
    wave_gan_model = WGAN(
        raw_maestro, [[-1, 2**14], [-1, 128, 256]], [[-1, 2**14, 1], [-1, 128, 256, 1]],
        generator, [discriminator, spec_discriminator], Z_DIM, generator_optimizer,
        discriminator_optimizer, generator_training_ratio=D_UPDATES_PER_G, batch_size=BATCH_SIZE,
        epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR, fn_compute_loss=_compute_losses,
        fn_save_examples=save_examples,
        get_discriminator_input_representations=_get_discriminator_input_representations)

    wave_gan_model.train()
    
if __name__ == '__main__':
    main()