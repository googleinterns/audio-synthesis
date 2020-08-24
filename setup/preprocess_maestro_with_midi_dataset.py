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
This scripts also extracts the aligned MIDI data.
The music chunks are then saved as a .npz file, same as
the MIDI data.
"""

import sys
import os
import glob
import librosa
import mido
import numpy as np
from audio_synthesis.setup import preprocessing_helpers

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14

BLOCK_IN_SECONDS = DATA_POINT_LENGTH / SAMPLE_RATE
NUM_STATES_PER_CHUNK = 512
HISTORICAL_CONDITIONING_BLOCKS = 1
TOTAL_CONDITIONING = (HISTORICAL_CONDITIONING_BLOCKS+1) * NUM_STATES_PER_CHUNK
RAW_DATA_PATH = './data/maestro/2017'
PROCESSED_DATA_PATH = './data/'

def main():
    audio_paths = glob.glob(RAW_DATA_PATH + '/**/*.wav', recursive=True)

    # Load audio files until we reach our desired
    # data set size.
    data = []
    midi_data = []
    hours_loaded = 0
    for audio_file_path in audio_paths:
        print(audio_file_path)
        wav, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)

        hours_loaded += len(wav) / SAMPLE_RATE / 60 / 60
        if hours_loaded >= APPROX_TOTAL_HOURS:
            break

        waveform_chunks, padded_wav_length = preprocessing_helpers.waveform_2_chunks(
            wav, DATA_POINT_LENGTH
        )

        midi_file_path = audio_file_path[:-3] + 'midi'
        midi = mido.MidiFile(midi_file_path, clip=True)

        processed_midi = preprocessing_helpers.piano_midi_2_chunks(
            midi, padded_wav_length, DATA_POINT_LENGTH, NUM_STATES_PER_CHUNK, SAMPLE_RATE
        ).astype(np.float32)

        midi_conditioning = []
        for chunk_idx in range(processed_midi.shape[0]):
            conditioning = processed_midi[
                max(chunk_idx - HISTORICAL_CONDITIONING_BLOCKS, 0) :chunk_idx+1, :, :
            ]
            conditioning = np.reshape(conditioning, (-1, 89))
            conditioning = np.pad(
                conditioning,
                [[TOTAL_CONDITIONING - conditioning.shape[0], 0], [0, 0]]
            )

            conditioning = np.expand_dims(conditioning, 0)
            midi_conditioning.extend(conditioning.astype(np.float32))

        midi_conditioning = np.array(midi_conditioning)

        assert waveform_chunks.shape[0] == processed_midi.shape[0]
        data.extend(waveform_chunks)
        midi_data.extend(midi_conditioning)

    data = np.array(data)
    midi_data = np.array(midi_data)

    print('Dataset Stats:')
    print('Total Hours: ', len(data) / 60 / 60)
    print('Dataset Size: ', sys.getsizeof(data) / 1e9, 'GB')
    print('Dataset Shape: ', data.shape)

    print("Saving Waveform Dataset")
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(data))
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_midi_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(midi_data))

if __name__ == '__main__':
    main()
