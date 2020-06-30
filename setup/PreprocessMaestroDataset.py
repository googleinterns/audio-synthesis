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

import librosa
import scipy.signal
import soundfile as sf
import sys
import os 
from os import listdir
from os.path import isfile, join
import math
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
raw_data_path = '../data/maestro/2017'
processed_data_path = '../data/'
## ##

OVERLAP = int(SEGMENT_LENGTH * OVERLAP_PERCENTANGE) 


def listdir(path, files=[]):
    if not os.path.isdir(path):
        if path.endswith('wav'):
            files.append(path)
        
    else:
        for item in os.listdir(path):
            listdir((path + '/' + item) if not path == '/' else '/' + item, files) 

    return files

files = listdir(raw_data_path)

raw_audio = []
hours_loaded = 0
for file in files:
    print(file)
    wav, _ = librosa.load(file, sr=SAMPLE_RATE)
    raw_audio.append(wav)
    
    song_length = ((len(wav) / SAMPLE_RATE) / 60) / 60
    hours_loaded += song_length
    
    if hours_loaded >= APPROX_TOTAL_HOURS:
        break
        

X = []
for wav in raw_audio:
    for i in range(0, len(wav), DATA_POINT_LENGTH):
        x = wav[i:i+DATA_POINT_LENGTH]
        
        # We might need to add padding on the right for the
        # last block
        if len(x) < DATA_POINT_LENGTH:
            x = np.pad(x, [[0, DATA_POINT_LENGTH-len(x)]])
        
        X.append(x)
        
X = np.array(X)

print('Dataset Stats:')
print('Total Hours: ', (len(X) / 60) / 60)
print('Dataset Size: ', sys.getsizeof(X) / (1e9))
print('Dataset Shape: ', X.shape)

print("Saving Waveform Dataset")
np.savez_compressed(processed_data_path + 'MAESTRO_' + str(APPROX_TOTAL_HOURS) + 'h.npz', np.array(X))