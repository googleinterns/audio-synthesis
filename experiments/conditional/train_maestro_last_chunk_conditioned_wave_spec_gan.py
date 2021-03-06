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

"""Training script for the last second (i.e., historically)
conditioned WaveSpecGAN Model.
"""

import os
import tensorflow as tf
from audio_synthesis.structures import ls_conditional_wave_spec_gan
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.models import conditional_wgan
from audio_synthesis.utils import waveform_save_helper as save_helper
from audio_synthesis.utils import spectral

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

_EPSILON = 1e-6
D_UPDATES_PER_G = 5
Z_DIM = 64
BATCH_SIZE = 32
EPOCHS = 300
SAMPLE_RATE = 16000
FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
SIGNAL_LENGTH = 2**14
WAVEFORM_SHAPE = [-1, SIGNAL_LENGTH, 1]
MAGNITUDE_IMAGE_SHAPE = [-1, 128, 256, 1]
CRITIC_WEIGHTINGS = [1.0, 1.0/1000.0]
CHECKPOINT_DIR = '_results/conditioning/LSC_WaveSpecGAN_HR_8192/training_checkpoints/'
RESULT_DIR = '_results/conditioning/LSC_WaveSpecGAN_HR_8192/audio/'
MAESTRO_PATH = 'data/MAESTRO_ls_6h.npz'
MAESTRO_CONDITIONING_PATH = 'data/MAESTRO_ls_hlf_cond_6h.npz'

def _get_discriminator_input_representations(x_in):
    """Computes the input representations for the WaveSpecGAN discriminator models,
    returning the input waveform and coresponding spectogram representations

    Args:
        x_in: A batch of waveforms with shape (-1, SIGNAL_LENGTH).

    Returns:
        A tuple containing the waveform and spectogram representaions of
        x_in.
    """

    magnitude = spectral.waveform_2_magnitude(
        x_in, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP, log_magnitude=True
    )
    magnitude = tf.expand_dims(magnitude, axis=3)

    return (x_in, magnitude)

def main():
    raw_maestro = waveform_dataset.get_waveform_dataset(MAESTRO_PATH)
    raw_maestro_conditioning = waveform_dataset.get_waveform_dataset(MAESTRO_CONDITIONING_PATH)

    generator = ls_conditional_wave_spec_gan.Generator()
    discriminator = ls_conditional_wave_spec_gan.WaveformDiscriminator(
        input_shape=WAVEFORM_SHAPE, weighting=CRITIC_WEIGHTINGS[0]
    )
    spec_discriminator = ls_conditional_wave_spec_gan.SpectogramDiscriminator(
        input_shape=MAGNITUDE_IMAGE_SHAPE, weighting=CRITIC_WEIGHTINGS[1]
    )

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    get_waveform = lambda waveform: waveform

    save_examples = lambda epoch, real, generated:\
        save_helper.save_wav_data(
            epoch, real, generated, SAMPLE_RATE, RESULT_DIR, get_waveform
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
