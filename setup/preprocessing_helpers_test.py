# Lint as: python3
"""Tests for preprocessing helpers."""

import os
import mido
import tensorflow as tf
import numpy as np

import preprocessing_helpers

TEMPO = 500000 # microseconds per beat
KEY_OFFSET = 21 # Piano notes start from midi node index 22.
MAX_KEY_VELOCITY = 127
N_PIANO_KEYS = 88

class LayersTest(tf.test.TestCase):

    def test_waveform_2_chunks_no_padding(self):
        signal_length = 128
        chunk_length = 16
        waveform = np.arange(0, signal_length)
        chunks, padded_length = preprocessing_helpers.waveform_2_chunks(waveform, chunk_length)

        self.assertEqual(chunks.shape, (signal_length // chunk_length, chunk_length))
        self.assertEqual(padded_length, signal_length)
        self.assertAllEqual(chunks[0], np.arange(chunk_length))

    def test_waveform_2_chunks_with_padding(self):
        signal_length = 120
        expected_padded_signal_length = 128
        chunk_length = 16
        waveform = np.arange(0, signal_length)
        chunks, padded_length = preprocessing_helpers.waveform_2_chunks(waveform, chunk_length)

        self.assertEqual(padded_length, expected_padded_signal_length)
        self.assertEqual(
            chunks.shape,
            (expected_padded_signal_length // chunk_length, chunk_length)
        )

    def test_midi_2_absolute_time(self):
        n_messages = 10
        track = mido.MidiTrack()

        for i in range(n_messages):
            track.append(mido.Message('note_on', note=64, velocity=64, time=1))

        preprocessing_helpers.midi_2_absolute_time(track)

        for i in range(n_messages):
            self.assertEqual(track[i].time, i+1)

    def test_piano_midi_2_chunks_key_detection(self):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        midi.tracks.append(track)

        for i in range(N_PIANO_KEYS):
            track.append(mido.Message('note_on', note=KEY_OFFSET+i, velocity=127, time=1))
        track.append(mido.Message('control_change', control=64, value=127, time=1))

        state = preprocessing_helpers.piano_midi_2_chunks(midi, 100, 100, 1, 100)
        self.assertAllEqual(np.reshape(state, -1), np.ones(N_PIANO_KEYS+1))

    def test_piano_midi_2_chunks_key_change_detection(self):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        midi.tracks.append(track)

        chunk_length_in_ticks = mido.second2tick(1, midi.ticks_per_beat, TEMPO)

        for i in range(N_PIANO_KEYS):
            track.append(
                mido.Message('note_on', note=KEY_OFFSET+i, velocity=127, time=0)
            )
            track.append(
                mido.Message('note_off', note=KEY_OFFSET+i,
                             velocity=0, time=chunk_length_in_ticks)
            )

        track.append(
            mido.Message('control_change', control=64, value=127, time=0)
        )
        track.append(
            mido.Message('control_change', control=64,
                         value=0, time=chunk_length_in_ticks)
        )

        state = preprocessing_helpers.piano_midi_2_chunks(midi, 89, 1, 1, 1)
        self.assertAllEqual(np.squeeze(state), np.diag(np.ones(N_PIANO_KEYS+1)))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.test.main()
