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


import numpy as np
import scipy.signal
import tensorflow as tf
from utils.Spectral import waveform_2_spectogram
    

def get_maestro_waveform_dataset(path):
    maestro = np.load(path)['arr_0']
    return maestro
    
    
def get_maestro_magnitude_phase_dataset(path, fft_length=512, frame_step=128, log_magnitude=True, instantaneous_frequency=True, mel_scale=False):
    maestro = get_maestro_waveform_dataset(path)
    
    def process_spectogram(x):
        return waveform_2_spectogram(x, fft_length=fft_length, frame_step=frame_step, log_magnitude=log_magnitude, instantaneous_frequency=instantaneous_frequency)
    
    maestro = np.array(list(map(process_spectogram, maestro)))
    return maestro
    