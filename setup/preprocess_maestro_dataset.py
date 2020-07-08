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
"""

import sys
import os
import librosa
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
RAW_DATA_PATH = '../data/maestro/2017'
PROCESSED_DATA_PATH = '../data/'

def list_wav_in_dir(path, files=[]): # pylint: disable=dangerous-default-value
    """Recursively collects all the '.wav' files
    in a directory.
    """

    if not os.path.isdir(path):
        if path.endswith('wav'):
            files.append(path)

    else:
        for item in os.listdir(path):
            list_wav_in_dir((path + '/' + item) if not path == '/' else '/' + item, files)

    return files

if __name__ == '__main__':
    audio_paths = list_wav_in_dir(RAW_DATA_PATH)

    # Load audio files until we reach our desired
    # data set size.
    raw_audio = []
    hours_loaded = 0 # pylint: disable=invalid-name
    for file in audio_paths:
        print(file)
        wav, _ = librosa.load(file, sr=SAMPLE_RATE)
        raw_audio.append(wav)

        song_length = ((len(wav) / SAMPLE_RATE) / 60) / 60
        hours_loaded += song_length

        if hours_loaded >= APPROX_TOTAL_HOURS:
            break

    # Split each song into chunks of size DATA_POINT_LENGTH
    data = []
    for wav in raw_audio:
        for i in range(0, len(wav), DATA_POINT_LENGTH):
            chunk = wav[i:i+DATA_POINT_LENGTH]

            # We might need to add padding on the right for the
            # last block
            if len(chunk) < DATA_POINT_LENGTH:
                chunk = np.pad(chunk, [[0, DATA_POINT_LENGTH-len(chunk)]])

            data.append(chunk)

    data = np.array(data)

    print('Dataset Stats:')
    print('Total Hours: ', (len(data) / 60) / 60)
    print('Dataset Size: ', sys.getsizeof(data) / (1e9))
    print('Dataset Shape: ', data.shape)

    print("Saving Waveform Dataset")
    np.savez_compressed(PROCESSED_DATA_PATH + 'MAESTRO_'\
                        + str(APPROX_TOTAL_HOURS) + 'h.npz',
                        np.array(data))
