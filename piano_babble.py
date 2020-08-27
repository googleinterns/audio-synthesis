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

"""
"""

import os
from tensorflow.keras import utils
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral
from audio_synthesis.structures import conditional_spec_gan

FFT_FRAME_LENGTH = 512
FFT_FRAME_STEP = 128
MAESTRO_PATH = 'data/MAESTRO_ls_hlf_cond_6h.npz'

def main():
    # Set allowed GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    raw_maestro_conditioning = maestro_dataset.get_maestro_stft_dataset(
        MAESTRO_PATH, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP
    )

    generator = conditional_spec_gan.Generator(channels=2, in_shape=(4, 8, 512))

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
    
    N_GENERATIONS = 60
    sequence = []
    for i in range(N_GENERATIONS):
        print(seed.shape)
        z_in = tf.random.uniform((1, 64), -1, 1)
        gen = generator(seed, z_in)
        print(gen.shape)
        wav = get_waveform(gen)
        wav = wav[0:2**14]
        print(wav.shape)
        sequence.append(np.reshape(wav, (2**14)))
        wav_cond = wav[2**13:2**14]
        seed = np.expand_dims(get_stft(wav_cond), 0)
     
    audio = np.array(sequence)
    audio = np.squeeze(audio)
    audio = np.reshape(audio, (-1))
    sf.write('stftmaggan_babble.wav', audio, 16000)
    
if __name__ == '__main__':
    main()