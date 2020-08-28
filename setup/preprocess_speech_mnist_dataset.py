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


"""Preprocesses the SpeechCommands dataset to extract SpeechMNIST.

Extracts the utterances of the numbers zero through nine. Adds a small
amount of padding to make te data an even power of two.
Operates on the following file:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
"""

import sys
import os
import glob
import tqdm
import librosa
import numpy as np
from audio_synthesis.setup import preprocessing_helpers

SAMPLE_RATE = 16000
PADDED_DATA_POINT_LENGTH = 2**14
RAW_DATA_PATH = './data/speechcommands/'
PROCESSED_DATA_PATH = './data/'
SUB_FOLDERS = ['zero', 'one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine']
LIMIT_PER_FOLDER = 1850

def main():
    waveforms = []
    for folder in SUB_FOLDERS:
        folder_path = os.path.join(RAW_DATA_PATH, folder)
        loaded_paths = glob.glob(os.path.join(folder_path, '*.wav'))
        np.random.shuffle(loaded_paths)
        loaded_paths = loaded_paths[:LIMIT_PER_FOLDER]

        for audio_file_path in tqdm.tqdm(loaded_paths):
            wav, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)

            wav, _ = preprocessing_helpers.waveform_2_chunks(
                wav, PADDED_DATA_POINT_LENGTH
            )

            waveforms.extend(wav)

    waveforms = np.array(waveforms)

    print('Dataset Stats:')
    print('Total Hours: ', len(waveforms) / 60 / 60)
    print('Dataset Size: ', sys.getsizeof(waveforms) / 1e9, 'GB')
    print('Dataset Shape: ', waveforms.shape)

    print("Saving Waveform Dataset")
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'SpeechMNIST_{}.npz'\
                        .format(LIMIT_PER_FOLDER)),
                        np.array(waveforms))

if __name__ == '__main__':
    main()
