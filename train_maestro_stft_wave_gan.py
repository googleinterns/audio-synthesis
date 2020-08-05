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

"""Training Script for STFTWaveGAN on MAESTRO.

Follows the same setup as SpecPhaseGAN, but
generates STFTs instead of Magnitude and Instantaneous
Frequency. Additionally, a Waveform critic is added
"""

import os
import tensorflow as tf
from audio_synthesis.structures import spec_gan, wave_gan
from audio_synthesis.models import wgan
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import maestro_save_helper as save_helper
from audio_synthesis.utils import spectral

# Setup Paramaters
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
Z_IN_SHAPE = [4, 8, 1024]
SPECTOGRAM_IMAGE_SHAPE = [-1, 128, 256, 2]
SIGNAL_LENGTH = 2**14
WAVEFORM_SHAPE = [-1, SIGNAL_LENGTH, 1]
CRITIC_WEIGHTINGS = [1.0, 1.0/1000.0]
CHECKPOINT_DIR = '_results/representation_study/STFTWaveGAN_HR/training_checkpoints/'
RESULT_DIR = '_results/representation_study/STFTWaveGAN_HR/audio/'
MAESTRO_PATH = 'data/MAESTRO_6h.npz'

def _get_discriminator_input_representations(stft_in):
    """Computes the input representations for the STFTWaveGAN discriminator models,
    returning the input waveform and coresponding spectogram representations

    Args:
        x_in: A batch of waveforms with shape (-1, SIGNAL_LENGTH).

    Returns:
        A tuple containing the waveform and spectogram representaions of
        x_in.
    """

    stft_in = tf.squeeze(stft_in)

    waveform = spectral.stft_2_waveform(
        stft_in, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )
    waveform = tf.squeeze(waveform)
    waveform = waveform[:, 0:SIGNAL_LENGTH]

    return (stft_in, waveform)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = maestro_dataset.get_maestro_stft_dataset(
        MAESTRO_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )
    print(raw_maestro.shape)

    generator = spec_gan.Generator(channels=2, in_shape=Z_IN_SHAPE)
    discriminator = spec_gan.Discriminator(input_shape=SPECTOGRAM_IMAGE_SHAPE, weighting=CRITIC_WEIGHTINGS[1])
    waveform_discriminator = wave_gan.Discriminator(input_shape=WAVEFORM_SHAPE, weighting=CRITIC_WEIGHTINGS[0])

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

    spec_phase_gan_model = wgan.WGAN(
        raw_maestro, generator, [discriminator, waveform_discriminator], Z_DIM,
        generator_optimizer, discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR, fn_save_examples=save_examples,
        fn_get_discriminator_input_representations=_get_discriminator_input_representations
    )

    spec_phase_gan_model.train()

if __name__ == '__main__':
    main()
