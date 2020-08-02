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
from audio_synthesis.structures import conditional_wave_spec_gan

MAESTRO_PATH = 'data/MAESTRO_ls_cond_6h.npz'

def main():
    # Set allowed GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)

    generator = conditional_wave_spec_gan.Generator()

    checkpoint_path = '_results/conditioning/LSC_WaveSpecGAN_HR_8192/training_checkpoints/ckpt-2'

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    
    seed = raw_maestro[100]
    N_GENERATIONS = 60
    sequence = [np.reshape(seed, (2**14))]
    for i in range(N_GENERATIONS):
        z_in = tf.random.uniform((1, 64), -1, 1)
        seed_in = sequence[i][2**13:]
        seed_in = np.expand_dims(seed_in, 0)
        gen = generator(seed_in, z_in)
        sequence.append(np.reshape(gen, (2**14)))
     
    audio = np.array(sequence)
    audio = np.squeeze(audio)
    audio = np.reshape(audio, (-1))
    sf.write('babble.wav', audio, 16000)
    
if __name__ == '__main__':
    main()