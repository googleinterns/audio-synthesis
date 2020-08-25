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

"""Training Script for WaveSpecGAN on a waveform dataset.
"""

import os
import tensorflow as tf
from audio_synthesis.structures import wave_gan
from audio_synthesis.structures import spec_gan
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.models import wgan
from audio_synthesis.utils import waveform_save_helper as save_helper
from audio_synthesis.utils import spectral

D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 64
EPOCHS = 300
SAMPLING_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
MEL_SPECTROGRAM = False
MEL_LOWER_HERTZ_EDGE = 80.
MEL_UPPER_HERTZ_EDGE = 7600.
NUM_MEL_BINS = 96
SIGNAL_LENGTH = 2**14
WAVEFORM_SHAPE = [-1, SIGNAL_LENGTH, 1]
CRITIC_WEIGHTINGS = [1.0, 1.0/1000.0]
CHECKPOINT_DIR = '_results/representation_study/SpeechMNIST/WaveMagGAN_HR/training_checkpoints/'
RESULT_DIR = '_results/representation_study/SpeechMNIST/WaveMagGAN_HR/audio/'
DATASET_PATH = 'data/SpeechMNIST_1850.npz'

if MEL_SPECTROGRAM:
    MAGNITUDE_IMAGE_SHAPE = [-1, 128, NUM_MEL_BINS, 1]
else:
    MAGNITUDE_IMAGE_SHAPE = [-1, 128, 256, 1]

def _get_discriminator_input_representations(x_in):
    """Computes the input representations for the WaveSpecGAN discriminator models,
    returning the input waveform and coresponding spectogram representations

    Args:
        x_in: A batch of waveforms with shape (-1, SIGNAL_LENGTH).

    Returns:
        A tuple containing the waveform and spectogram representaions of
        x_in.
    """

    x_in = tf.squeeze(x_in)

    if MEL_SPECTROGRAM:
        magnitude = spectral.waveform_2_magnitude(
            x_in, FFT_FRAME_LENGTH, FFT_FRAME_STEP, True,
            NUM_MEL_BINS, MEL_LOWER_HERTZ_EDGE, MEL_UPPER_HERTZ_EDGE
        )
    else:
        magnitude = spectral.waveform_2_magnitude(
            x_in, FFT_FRAME_LENGTH, FFT_FRAME_STEP, True
        )

    return (x_in, magnitude)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_dataset = waveform_dataset.get_waveform_dataset(DATASET_PATH)

    generator = wave_gan.Generator()
    discriminator = wave_gan.Discriminator(
        input_shape=WAVEFORM_SHAPE, weighting=CRITIC_WEIGHTINGS[0]
    )
    spec_discriminator = spec_gan.Discriminator(
        input_shape=MAGNITUDE_IMAGE_SHAPE, weighting=CRITIC_WEIGHTINGS[1]
    )

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda waveform: waveform

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLING_RATE,
            RESULT_DIR, get_waveform
        )

    wave_gan_model = wgan.WGAN(
        raw_dataset, generator, [discriminator, spec_discriminator], Z_DIM, generator_optimizer,
        discriminator_optimizer, discriminator_training_ratio=D_UPDATES_PER_G,
        batch_size=BATCH_SIZE, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR, fn_save_examples=save_examples,
        fn_get_discriminator_input_representations=_get_discriminator_input_representations
    )

    wave_gan_model.restore('ckpt-14', 140)
    wave_gan_model.train()

if __name__ == '__main__':
    main()
