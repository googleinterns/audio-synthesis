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
from matplotlib import pyplot as plt
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.utils import spectral
from audio_synthesis.structures import conditional_wave_spec_gan

MAESTRO_PATH = 'data/MAESTRO_ls_6h.npz'
MAESTRO_CONDITIONING_PATH = 'data/MAESTRO_ls_hlf_cond_6h.npz'

BATCH_SIZE = 100

def main():
    # Set allowed GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)
    raw_maestro_conditioning = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_CONDITIONING_PATH)

    generator = conditional_wave_spec_gan.Generator()

    checkpoint_path = '_results/conditioning/LSC_WaveSpecGAN_HR_8192/training_checkpoints/ckpt-30'

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    
    
    generations = None
    pb_i = utils.Progbar(len(raw_maestro_conditioning))
    for i in range(0, len(raw_maestro_conditioning), BATCH_SIZE):
        x_batch = raw_maestro_conditioning[i:i+BATCH_SIZE]
        z_in = tf.random.uniform((x_batch.shape[0], 64), -1, 1)
        
        x_gen = generator(x_batch, z_in)
        x_gen = tf.squeeze(x_gen)
        if generations is None:
            generations = x_gen
        else:
            generations = np.append(generations, x_gen, axis=0)
        pb_i.add(BATCH_SIZE)
        
    #generations = np.array(generations)
    #generations = np.squeeze(generations)
    print(generations.shape)
    
    
    raw_maestro_conditioning_norm = np.linalg.norm(raw_maestro_conditioning, axis=-1)
    generations_norm = np.linalg.norm(generations, axis=-1)

    plt.hist(generations_norm, bins=100)
    plt.savefig('generation_norm_histogram.png')
    plt.clf()

    plt.hist(raw_maestro_conditioning_norm, bins=100)
    plt.savefig('raw_maestro_conditioning_norm_histogram.png')
    plt.clf()
    
    THRESHOLD = 1.0
    generations_silent = generations_norm < THRESHOLD
    
    silent_data_wav = generations[generations_silent][0:100]
    silent_data_wav = np.reshape(silent_data_wav, (-1,))
    sf.write('silent_generations.wav', silent_data_wav, 16000)
    
    silent_conditioning_wav = raw_maestro_conditioning[generations_silent][0:100]
    silent_conditioning_wav = np.reshape(silent_conditioning_wav, (-1,))
    sf.write('silent_conditioning.wav', silent_conditioning_wav, 16000)
    
    
if __name__ == '__main__':
    main()