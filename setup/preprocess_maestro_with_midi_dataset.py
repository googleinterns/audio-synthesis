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

import os
import sys
import glob
import copy
import librosa
import mido
import numpy as np

# Overall Config #
APPROX_TOTAL_HOURS = 6
SAMPLE_RATE = 16000 # 16kHz
DATA_POINT_LENGTH = 2**14
N_PIANO_KEYS = 88
TEMPO = 500000 # microseconds per beat
BLOCK_IN_SECONDS = DATA_POINT_LENGTH / SAMPLE_RATE
BLOCK_N_QUANT = 128
BLOCK_ZONE_LENGTH = BLOCK_IN_SECONDS / BLOCK_N_QUANT
RAW_DATA_PATH = './data/maestro/2017'
PROCESSED_DATA_PATH = './data/'

def waveform_2_chunks(wavform):
    """Converts a given waveform into chunks of a pre-defined
    size. Padds the waveform to ensure a whole number of chunks.

    Args:
        waveform: The waveform to be processed. Expected
            shape is [time]
    """

    # Pad the song to ensure it can be evenly divided into
    # chunks of length 'DATA_POINT_LENGTH'
    padded_wav_length = int(DATA_POINT_LENGTH * np.ceil(len(wavform) / DATA_POINT_LENGTH))
    wavform = np.pad(wavform, [[0, padded_wav_length - len(wavform)]])

    chunks = np.reshape(wavform, (-1, DATA_POINT_LENGTH))
    return chunks, padded_wav_length

def midi_2_absolute_time(midi_track):
    """Converts a relative time MIDI track to
    absolute time.

    Args:
        midi_track: The relative time MIDI track to
            be processed.
    """

    total_time = 0
    for msg in midi_track:
        total_time += msg.time
        msg.time = total_time

def midi_2_chunks(mid, padded_wav_length):
    """Function handles pre-processing the event based MIDI data
    for a given recording.

    In this processing, we convert the event based MIDI representation
    into a descrete block format that can be provided as input to our
    neural network models. The processing takes the form of the folowing steps:
    1) Convert the relative time to absolute time
    2) Process the midi events for each second, corrosponding to a given
       second of music data. Each second consists of BLOCK_N_QUANT state vectors,
       where the state is the velocity of the keys in that quantized time region.
       We also keep track of the duration of each note press, this information is
       contained in another channel.

    Args:
        mid: The loaded midi data file
        padded_wav_length: The padded length of the waveform that
            coresponds to the midi data.

    Returns:
        song_midi_time: The velocity information loaded from the MIDI data
            and time information that linearly decays from 1 to 0, one being
            the note start.
    """

    track = mid.tracks[1]
    midi_2_absolute_time(track)

    # Account for padding in the WAV file.
    total_time = mido.second2tick(padded_wav_length // SAMPLE_RATE, mid.ticks_per_beat, TEMPO)
    block_in_ticks = mido.second2tick(BLOCK_IN_SECONDS, mid.ticks_per_beat, TEMPO)

    # Split the total number of MIDI ticks into blocks of time that
    # correspond to the second long chunks of audio extracted. Each values
    # is the end point of a zone.
    time_chunks = np.arange(0, total_time, block_in_ticks, dtype=np.float32)[1:]
    time_chunks = np.append(time_chunks, total_time)

    # Compute the quantized zones within each block.
    block_zones = np.arange(-block_in_ticks, 0, block_in_ticks / (BLOCK_N_QUANT))[1:]
    block_zones = np.append(block_zones, 0)

    all_zones = np.reshape(list(map(lambda end_point: end_point + block_zones, time_chunks)), (-1,))

    song = []
    song_time = []
    note_start = np.zeros((N_PIANO_KEYS)).astype(np.int32)
    times = np.zeros((N_PIANO_KEYS))
    state = np.zeros((N_PIANO_KEYS))
    event_idx = 0
    time_idx = 0
    # For each quantied zone in the block, update the state information,
    # with the events that occour in the zone and append the state to the block.
    for zone in all_zones:
        # Each block zone, we want to increment non-zero times by one
        times = [times[i] + 1 if not times[i] == 0 else 0 for i in range(len(times))]

        while event_idx < len(track) and track[event_idx].time < zone:
            note_type = track[event_idx].type
            if not note_type == 'note_on' and not note_type == 'note_off':
                event_idx += 1
                continue

            note_idx = track[event_idx].note - 21
            note_velocity = track[event_idx].velocity

            # If note is released
            if note_velocity == 0:
                state[note_idx] = 0

                # Iterate through the time information, normalizing so it sits between
                # 0 and 1.
                diff = (time_idx - note_start[note_idx])
                for i in range(note_start[note_idx], time_idx):
                    song_time[i][note_idx] = 1.0 - (song_time[i][note_idx] / diff)

                times[note_idx] = 0
                note_start[note_idx] = 0
            else:
                state[note_idx] = (note_velocity / 127)
                note_start[note_idx] = time_idx
                times[note_idx] = 1

            event_idx += 1
        time_idx += 1
        song.append(copy.deepcopy(state))
        song_time.append(copy.deepcopy(times))

    song = np.reshape(song, (-1, BLOCK_N_QUANT, 88))
    song_time = np.reshape(song_time, (-1, BLOCK_N_QUANT, 88))

    song_midi_time = np.concatenate(
        [np.expand_dims(song, 3), np.expand_dims(song_time, 3)],
        axis=-1
    )

    return song_midi_time

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

        waveform_chunks, padded_wav_length = waveform_2_chunks(wav)
        data.extend(waveform_chunks)

        midi_file_path = audio_file_path[0:-3] + 'midi'
        mid = mido.MidiFile(midi_file_path, clip=True)
        
        processed_midi = midi_2_chunks(mid, padded_wav_length)
        midi_data.extend(processed_midi)

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