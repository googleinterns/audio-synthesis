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
last second conditioning.
"""

import os
import tensorflow as tf
import soundfile as sf
import numpy as np
from audio_synthesis.datasets import waveform_dataset
from audio_synthesis.structures import ls_conditional_wave_spec_gan

MAESTRO_PATH = 'data/MAESTRO_ls_cond_6h.npz'
N_GENERATIONS = 60
SEED_INDEX = 100
SAMPLE_RATE = 16000
CONDITIONING_START_INDEX = 2**13
GENERATION_LENGTH = 2**14
Z_DIM = 64

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = waveform_dataset.get_waveform_dataset(MAESTRO_PATH)

    generator = ls_conditional_wave_spec_gan.Generator()

    checkpoint_path = '_results/conditioning/LSC_WaveSpecGAN_HR_8192/training_checkpoints/ckpt-2'

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()

    seed = raw_maestro[SEED_INDEX]

    sequence = [np.reshape(seed, (GENERATION_LENGTH))]
    for i in range(N_GENERATIONS):
        z_in = tf.random.uniform((1, Z_DIM), -1, 1)
        seed_in = sequence[i][CONDITIONING_START_INDEX:]
        seed_in = np.expand_dims(seed_in, 0)
        
        gen = generator(seed_in, z_in)
        sequence.append(np.squeeze(gen))

    audio = np.reshape(sequence, (-1))
    sf.write('babble.wav', audio, SAMPLE_RATE)

if __name__ == '__main__':
    main()
