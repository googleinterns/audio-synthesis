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
import mido
import bisect
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
BLOCK_IN_SECONDS = DATA_POINT_LENGTH / SAMPLE_RATE
BLOCK_N_QUANT = 128
BLOCK_ZONE_LENGTH = BLOCK_IN_SECONDS / BLOCK_N_QUANT
RAW_DATA_PATH = './data/maestro/2017'
PROCESSED_DATA_PATH = '../data/'

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
        print(len(wav) / 16000)

        hours_loaded += len(wav) / SAMPLE_RATE / 60 / 60
        if hours_loaded >= APPROX_TOTAL_HOURS:
            break

        # Pad the song to ensure it can be evenly divided into
        # chunks of length 'DATA_POINT_LENGTH'
        padded_wav_length = int(DATA_POINT_LENGTH * np.ceil(len(wav) / DATA_POINT_LENGTH))
        wav = np.pad(wav, [[0, padded_wav_length - len(wav)]])

        chunks = np.reshape(wav, (-1, DATA_POINT_LENGTH))
        data.extend(chunks)
        
        # Load and process MIDI file
        midi_file_path = audio_file_path[0:-3] + 'midi'#.replace('.wav', '.midi')
        print(midi_file_path)
        mid = mido.MidiFile(midi_file_path, clip=True)
        track = mid.tracks[1]
        
        # Convert Track to Absolute time
        total_time = 0
        for msg in track:
            total_time += msg.time
            msg.time = total_time#mido.tick2second(total_time, mid.ticks_per_beat, 500000)
        # Account for padding?
        print(total_time)
        total_time = mido.second2tick(padded_wav_length // SAMPLE_RATE, mid.ticks_per_beat, 500000)
        print(total_time)
        
        # Split into seconds and convert into array
        # of (time_steps, 88)
        start_idx = 0
        seconds_elapsed = 0.0
        current_time = 0.0
        end_idx = 1
        state = np.zeros(88)
        
        # Compute time chunks in 'tic-time'
        BLOCK_IN_TICKS = mido.second2tick(BLOCK_IN_SECONDS, mid.ticks_per_beat, 500000)
        time_chunks = np.arange(0, total_time, BLOCK_IN_TICKS, dtype=np.float32)
        time_chunks = np.append(time_chunks, total_time)
        
        
        print(len(time_chunks[1:]))
        print(len(chunks))
        for end_point in time_chunks[1:]:
            while end_idx < len(track) and track[end_idx].time < end_point:
                end_idx += 1
                continue
                
            # Lets generate the block size and fill it in
            block_zones = np.arange(end_point - BLOCK_IN_TICKS, end_point, BLOCK_IN_TICKS / (BLOCK_N_QUANT))[1:]
            
            block = np.zeros((len(block_zones)+1, 88))
            while start_idx < end_idx:
                if not track[start_idx].type == 'note_on':
                    start_idx += 1
                    continue
                    
                idx = bisect.bisect_left(block_zones, track[start_idx].time)
                block[idx][(track[start_idx].note - 21)] = (track[start_idx].velocity / 127)
                start_idx += 1

            midi_data.append(block)

    data = np.array(data)
    midi_data = np.array(midi_data)
    print(data.shape)
    print(midi_data.shape)

    print('Dataset Stats:')
    print('Total Hours: ', len(data) / 60 / 60)
    print('Dataset Size: ', sys.getsizeof(data) / 1e9, 'GB')
    print('Dataset Shape: ', data.shape)

    print("Saving Waveform Dataset")
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(data))

if __name__ == '__main__':
    main()
