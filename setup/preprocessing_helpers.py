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

"""Exports a collection of helper functions for loading
and preprocessing waveform and MIDI data.
"""

import copy
import mido
import numpy as np

KEY_OFFSET = 21 # Piano notes start from midi node index 21.
MAX_KEY_VELOCITY = 127
N_PIANO_KEYS = 88
N_STATE_SLOTS = N_PIANO_KEYS + 1 # An additional slot for the sustain pedal.
TEMPO = 500000 # microseconds per beat

def waveform_2_chunks(wavform, chunk_length):
    """Converts a given waveform into chunks of a pre-defined
    size. Padds the waveform to ensure a whole number of chunks.

    Args:
        waveform: The waveform to be processed. Expected
            shape is [time].
        chunk_length: The desired length (in samples) of the
            chunks the input waveform will be split into.

    Returns:
        chunks: The input waveform split into chunks of the
            desired size.  Expected shape is
            [ceil(time / chunk_length), chunk_length].
        padded_wav_length: The length of the input waveform after
            padding has been added.
    """

    # Pad the song to ensure it can be evenly divided into
    # chunks of length 'DATA_POINT_LENGTH'
    padded_wav_length = int(chunk_length * np.ceil(len(wavform) / chunk_length))
    wavform = np.pad(wavform, [[0, padded_wav_length - len(wavform)]])

    chunks = np.reshape(wavform, (-1, chunk_length))
    return chunks, padded_wav_length

def midi_2_absolute_time(midi_track):
    """Converts a relative time MIDI track to
    absolute time. Modifies the MIDI track in
    place.

    Args:
        midi_track: The relative time MIDI track to
            be processed.
    """

    total_time = 0
    for msg in midi_track:
        total_time += msg.time
        msg.time = total_time

def piano_midi_2_chunks(midi, padded_wav_length, chunk_length, num_states_per_chunk, sample_rate):
    """Function handles pre-processing the event based MIDI data
    for a given recording.

    In this processing, we convert the event based MIDI representation
    into a descrete chunk format that can be provided as input to our
    neural network models. The processing takes the form of the folowing steps:
    1) Convert the relative time to absolute time
    2) Process the midi events for each chunk. Each chunk is quantized into
       num_states_per_chunk state vectors. Each state vector represents the keys
       that are currently pressed after the quantized time window.

    Args:
        midi: The loaded midi data file
        padded_wav_length: The padded length (in samples) of the waveform that
            coresponds to the midi data.
        chunk_length: The length of each chunk (in samples).
        num_states_per_chunk: The number of quantization blocks per chunk.
        sample_rate: The sample rate of the audio (samples per second).

    Returns:
        An array of shape (-1, num_states_per_chunk, N_STATE_SLOTS) containing the
        state for each quantized time period.
    """

    track = midi.tracks[1]
    midi_2_absolute_time(track)

    chunk_length_in_seconds = chunk_length / sample_rate
    chunk_length_in_ticks = mido.second2tick(
        chunk_length_in_seconds, midi.ticks_per_beat, TEMPO
    )
    total_time_in_ticks = mido.second2tick(
        padded_wav_length / sample_rate, midi.ticks_per_beat, TEMPO
    )
    ticks_per_state = chunk_length_in_ticks / num_states_per_chunk
    
    end_tick_indicies = np.arange(
        ticks_per_state,
        total_time_in_ticks + ticks_per_state,
        ticks_per_state, dtype=np.float32
    )
    
    # Account for a small rounding issue at the end of songs.
    end_tick_indicies = end_tick_indicies[
        :round(total_time_in_ticks / float(ticks_per_state))
    ]

    states = []
    state = np.zeros((N_STATE_SLOTS))
    event_idx = 0

    # For each quantied zone in the chunk, update the state information,
    # with the events that occour in the zone and append the state to the record.
    for end_tick_index in end_tick_indicies:
        while event_idx < len(track) and track[event_idx].time < end_tick_index:
            note_type = track[event_idx].type

            # Handle Piano Key Press/Release.
            if note_type in ('note_on', 'note_off'):
                note_idx = track[event_idx].note - KEY_OFFSET
                note_velocity = track[event_idx].velocity

                # If note is released
                if note_velocity == 0:
                    state[note_idx] = 0
                else:
                    state[note_idx] = note_velocity / MAX_KEY_VELOCITY
            # Handle a change in the sustain pedal.
            elif note_type == 'control_change':
                state[-1] = track[event_idx].value / MAX_KEY_VELOCITY

            event_idx += 1
        states.append(copy.deepcopy(state))
    
    states = np.reshape(states, (-1, num_states_per_chunk, N_STATE_SLOTS))
    return states
