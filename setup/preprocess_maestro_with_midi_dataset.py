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
import copy
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
BLOCK_IN_SECONDS = DATA_POINT_LENGTH / SAMPLE_RATE
BLOCK_N_QUANT = 128
BLOCK_ZONE_LENGTH = BLOCK_IN_SECONDS / BLOCK_N_QUANT
RAW_DATA_PATH = './data/maestro/2017'
PROCESSED_DATA_PATH = './data/'

def main():
    audio_paths = glob.glob(RAW_DATA_PATH + '/**/*.wav', recursive=True)

    # Load audio files until we reach our desired
    # data set size.
    data = []
    midi_data = []
    time_data = []
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
        midi_file_path = audio_file_path[0:-3] + 'midi'
        print(midi_file_path)
        mid = mido.MidiFile(midi_file_path, clip=True)
        track = mid.tracks[1]
        
        # Convert Track to Absolute time
        total_time = 0
        for msg in track:
            total_time += msg.time
            msg.time = total_time
        
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
        print(BLOCK_IN_TICKS)
        time_chunks = np.arange(0, total_time, BLOCK_IN_TICKS, dtype=np.float32)
        time_chunks = np.append(time_chunks, total_time)
        
        
        print(len(time_chunks[1:]))
        print(len(chunks))
        block_zones = np.arange(-BLOCK_IN_TICKS, 0, BLOCK_IN_TICKS / (BLOCK_N_QUANT))[1:]
        block_zones = np.append(block_zones, 0)
        block_length = len(block_zones)+1
        
        time_idx = 0
        note_start = np.zeros((88)).astype(np.int32)
        times = np.zeros((88))
        state = np.zeros((88))
        song = []
        song_time = []
        for end_point in time_chunks[1:]:
            #block = []
            while end_idx < len(track) and track[end_idx].time < end_point:
                end_idx += 1
                continue
            
            for zone in block_zones:
                # Each block, we want to increment non-zero times by one?
                times = [times[i] + 1 if not times[i] == 0 else 0 for i in range(len(times))]
                #print(times)
                
                while start_idx < end_idx and track[start_idx].time < (end_point + zone):
                    if not track[start_idx].type == 'note_on' and not track[start_idx].type == 'note_off':
                        start_idx += 1
                        continue
                        
                    #print(track[start_idx])
                    if track[start_idx].velocity == 0:
                        state[(track[start_idx].note - 21)] = 0
                        diff = (time_idx - note_start[(track[start_idx].note - 21)])
                        for i in range(note_start[(track[start_idx].note - 21)], time_idx):
                            song_time[i][(track[start_idx].note - 21)] = 1.0 - (song_time[i][(track[start_idx].note - 21)] / diff)
                            
                        times[(track[start_idx].note - 21)] = 0
                        note_start[(track[start_idx].note - 21)] = 0
                    else:
                        state[(track[start_idx].note - 21)] = (track[start_idx].velocity / 127)
                        note_start[(track[start_idx].note - 21)] = time_idx
                        times[(track[start_idx].note - 21)] = 1
                        
                    start_idx += 1
                time_idx += 1
                song.append(copy.deepcopy(state))
                song_time.append(copy.deepcopy(times))
                
        song = np.reshape(song, (-1, BLOCK_N_QUANT, 88))
        song_time = np.reshape(song_time, (-1, BLOCK_N_QUANT, 88))
        
        song_midi_time = np.concatenate([np.expand_dims(song, 3), np.expand_dims(song_time, 3)], axis=-1)
        print(song_midi_time.shape)
        
        midi_data.extend(song_midi_time)
            
        #save_data = midi_data[0:15]
        #save_data = np.reshape(save_data, (-1, 88, 2))
            
        #img_data = save_data[:,:,0]
        #time_data = save_data[:,:,1]
        #print(img_data.shape)
        
        #import matplotlib.pyplot as plt
        #plt.imshow(np.transpose(img_data), origin='lower')
        #plt.savefig('NoteData.png', bbox_inches='tight', dpi=360)
        
        #plt.clf()
        #plt.imshow(np.transpose(time_data), origin='lower')
        #plt.savefig('TimeData.png', bbox_inches='tight', dpi=360)
        
        #sys.exit(0)

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
    np.savez_compressed(os.path.join(PROCESSED_DATA_PATH, 'MAESTRO_midi_{}h.npz'\
                        .format(APPROX_TOTAL_HOURS)),
                        np.array(midi_data))

if __name__ == '__main__':
    main()
