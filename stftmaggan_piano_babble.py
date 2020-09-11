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

"""Generates "babbling piano music" from a model trained with
last second conditioning. Specifically, this uses the underlying
STFTMagGAN.
"""

import os
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.utils import spectral
from audio_synthesis.structures import ls_conditional_spec_gan

FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
N_GENERATIONS = 60
SAMPLE_RATE = 16000
CONDITIONING_START_INDEX = 2**13
GENERATION_LENGTH = 2**14
MAESTRO_PATH = 'data/MAESTRO_ls_hlf_cond_6h.npz'

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro_conditioning = waveform_dataset.get_stft_dataset(
        MAESTRO_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )

    generator = ls_conditional_spec_gan.Generator(channels=2, in_shape=(4, 8, 512))

    checkpoint_path = '_results/conditioning/LSC_STFTMagGAN_HR_8192/training_checkpoints/ckpt-11'

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()

    get_waveform = lambda stft:\
        spectral.stft_2_waveform(
            stft, FFT_FRAME_LENGTH, FFT_FRAME_STEP
        )[0]
    get_stft = lambda waveform:\
        spectral.waveform_2_stft(
            waveform, FFT_FRAME_LENGTH, FFT_FRAME_STEP
        )[0]

    seed = np.expand_dims(raw_maestro_conditioning[5], 0)

    sequence = []
    for _ in range(N_GENERATIONS):
        z_in = tf.random.uniform((1, 64), -1, 1)
        gen = generator(seed, z_in)
        wav = get_waveform(gen)[0:GENERATION_LENGTH]

        sequence.append(wav)
        wav_cond = wav[CONDITIONING_START_INDEX:]
        seed = np.expand_dims(get_stft(wav_cond), 0)

    audio = np.array(sequence)
    audio = np.squeeze(audio)
    audio = np.reshape(audio, (-1))
    sf.write('stftmaggan_babble.wav', audio, 16000)

if __name__ == '__main__':
    main()
