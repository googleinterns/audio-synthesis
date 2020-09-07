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
from audio_synthesis.structures import conditional_spec_gan
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models import conditional_wgan as wgan
from audio_synthesis.utils import maestro_save_helper as save_helper
from audio_synthesis.utils import spectral

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

_EPSILON = 1e-6
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 32
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
SIGNAL_LENGTH = 2**14
WAVEFORM_SHAPE = [-1, SIGNAL_LENGTH, 1]
SPECTOGRAM_IMAGE_SHAPE = [-1, 128, 256, 2]
MAGNITUDE_IMAGE_SHAPE = [-1, 128, 256, 1]
CRITIC_WEIGHTINGS=[1.0, 1.0]
CHECKPOINT_DIR = '_results/conditioning/LSC_STFTMagGAN_HR/training_checkpoints/'
RESULT_DIR = '_results/conditioning/LSC_STFTMagGAN_HR/audio/'
MAESTRO_PATH = 'data/MAESTRO_ls_6h.npz'
MAESTRO_CONDITIONING_PATH = 'data/MAESTRO_ls_cond_6h.npz'

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
    raw_maestro = maestro_dataset.get_maestro_stft_dataset(
        MAESTRO_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )
    raw_maestro_conditioning = maestro_dataset.get_maestro_stft_dataset(
        MAESTRO_CONDITIONING_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )

    generator = conditional_spec_gan.Generator(channels=2)
    stft_discriminator = conditional_spec_gan.Discriminator(input_shape=SPECTOGRAM_IMAGE_SHAPE)
    mag_discriminator = conditional_spec_gan.Discriminator(input_shape=MAGNITUDE_IMAGE_SHAPE)

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
    
    wave_gan_model = wgan.WGAN(
        (raw_maestro, raw_maestro_conditioning), [SPECTOGRAM_IMAGE_SHAPE, MAGNITUDE_IMAGE_SHAPE],
        [(-1, 64, 256, 2), (-1, 64, 256, 1)], generator,
        [stft_discriminator, mag_discriminator], Z_DIM, generator_optimizer,
        discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, lambdas=CRITIC_WEIGHTINGS,
        checkpoint_dir=CHECKPOINT_DIR, fn_save_examples=save_examples,
        fn_get_discriminator_input_representations=_get_discriminator_input_representations
    )

    wave_gan_model.train()
    
if __name__ == '__main__':
    main()