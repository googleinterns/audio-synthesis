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

"""Pre-Processes the MAESTRO Dataset.

Pre-proceses the raw MAESTRO dataset into a data set
of music chunks with a predefined lenth and sampling rate.
The music chunks are then saved as a .npz file
"""

import os
import sys
import glob
import librosa
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
RAW_DATA_PATH = './data/maestro/2017'
PROCESSED_DATA_PATH = './data/'

def main():
    audio_paths = glob.glob(RAW_DATA_PATH + '/**/*.wav', recursive=True)

    # Load audio files until we reach our desired
    # data set size.
    data = []
    data_conditioning = []
    hours_loaded = 0
    for audio_file_path in audio_paths:
        print(audio_file_path)
        wav, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)

        hours_loaded += len(wav) / SAMPLE_RATE / 60 / 60
        if hours_loaded >= APPROX_TOTAL_HOURS:
            break

        # Pad the song to ensure it can be evenly divided into
        # chunks of length 'DATA_POINT_LENGTH'
        padded_wav_length = int(DATA_POINT_LENGTH * np.ceil(len(wav) / DATA_POINT_LENGTH))
        wav = np.pad(wav, [[0, padded_wav_length - len(wav)]])

        chunks = np.reshape(wav, (-1, DATA_POINT_LENGTH))
        print(chunks.shape)
        
        conditioning_chunks = chunks[0:-1]
        conditioning_chunks = conditioning_chunks[:, (2**14//2):]
        chunks = chunks[1:]
        
        print(conditioning_chunks.shape)
        print(chunks.shape)
        
        data.extend(chunks)
        data_conditioning.extend(conditioning_chunks)


    data = np.array(data)
    data_conditioning = np.array(data_conditioning)
    
    print('Dataset Stats:')
    print('Total Hours: ', len(data) / 60 / 60)
    print('Dataset Size: ', sys.getsizeof(data) / 1e9, 'GB')
    print('Dataset Shape: ', data.shape)

    print("Saving Waveform Dataset")
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_ls_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(data))
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_ls_hlf_cond_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(data_conditioning))

if __name__ == '__main__':
    main()
