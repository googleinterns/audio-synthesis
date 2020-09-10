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

"""Training script for the MIDI conditioned STFTMagGAN Model."""

import os
import tensorflow as tf
import numpy as np
from audio_synthesis.structures import midi_conditional_spec_gan
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.models import conditional_wgan
from audio_synthesis.utils import waveform_save_helper as save_helper
from audio_synthesis.utils import spectral

D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 16
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
SIGNAL_LENGTH = 2**14
WAVEFORM_SHAPE = [-1, SIGNAL_LENGTH, 1]
STFT_IMAGE_SHAPE = [-1, 128, FFT_FRAME_LENGTH // 2, 2]
MAGNITUDE_IMAGE_SHAPE = [-1, 128, FFT_FRAME_LENGTH // 2, 1]
CRITIC_WEIGHTINGS = [1.0, 1.0 / 1000.0]
CHECKPOINT_DIR = '_results/midi_conditional/ConditionalSTFTMagGAN/training_checkpoints/'
RESULT_DIR = '_results/midi_conditional/ConditionalSTFTMagGAN/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h_old.npz'
MAESTRO_MIDI_PATH = 'data/MAESTRO_midi_6h_old.npz'

def _get_discriminator_input_representations(stft_in):
    """Computes the input representations for the STFTSpecGAN discriminator models,
    returning the input waveform and coresponding spectogram representations

    Args:
        x_in: A batch of stft with shape (-1, time, frequency_dims).

    Returns:
        A tuple containing the stft and log magnitude spectrum representaions of
        x_in.
    """
    
    real = stft_in[:, :, :, 0]
    imag = stft_in[:, :, :, 1]
    magnitude = tf.sqrt(tf.square(real) + tf.square(imag))
    magnitude = tf.math.log(magnitude + 1e-6)

    return (stft_in, magnitude)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = waveform_dataset.get_stft_dataset(
        MAESTRO_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    ).astype(np.float32)
    raw_maestro_conditioning = waveform_dataset.get_waveform_dataset(
        MAESTRO_MIDI_PATH).astype(np.float32)

    generator = midi_conditional_spec_gan.Generator(channels=2)
    discriminator = midi_conditional_spec_gan.Discriminator(
        input_shape=STFT_IMAGE_SHAPE
    )
    spec_discriminator = midi_conditional_spec_gan.Discriminator(
        input_shape=MAGNITUDE_IMAGE_SHAPE
    )

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda stft:\
        spectral.stft_2_waveform(
            stft, FFT_FRAME_LENGTH, FFT_FRAME_STEP
        )[0]

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE, RESULT_DIR, get_waveform
        )

    wave_gan_model = conditional_wgan.ConditionalWGAN(
        raw_maestro, raw_maestro_conditioning, generator, [discriminator, spec_discriminator],
        Z_DIM, generator_optimizer, discriminator_optimizer,
        discriminator_training_ratio=D_UPDATES_PER_G, batch_size=BATCH_SIZE, epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR, fn_save_examples=save_examples,
        fn_get_discriminator_input_representations=_get_discriminator_input_representations
    )

    wave_gan_model.train()

if __name__ == '__main__':
    main()
