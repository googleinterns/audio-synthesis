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

"""Exports the learned basis function decomposition experiemnt
that uses an auxiliary classification loss to encourage something
meaningful. This classification loss looks at predicting the MIDI
representation (piano keys pressed and the velocity + the sustain
pedal) from the decomposed signal.
"""

import os
import tensorflow as tf
from audio_synthesis.datasets import maestro_dataset
from audio_synthesis.models import learned_basis_decomposition
from audio_synthesis.structures import learned_basis_function

FILTER_LENGTH = 32
NUMBER_OF_FILTERS = 512
BATCH_SIZE = 64
EPOCHS = 100
N_MIDI_KEYS = 89
MAESTRO_PATH = 'data/MAESTRO_6h.npz'
MAESTRO_MIDI_PATH = 'data/MAESTRO_midi_512_6h.npz'
CHECKPOINT_DIR = '_results/learned_decomposition/classifier/training_checkpoints/'
RESULTS_DIR = '_results/learned_decomposition/classifier/audio/'


def compute_auxiliary_loss(num_steps, x_in, decomposition, x_hat,
                           auxiliary_models, auxiliary_optimizers):
    """Computes the auxiliary classification loss for the learned
    decomposition model. Follows the specification given in the file
    'models/learned_basis_decomposition.py'

    Args:
        num_steps: The number of training steps past this epoch.
        x_in: The batch of training data, a tuple, (signal, midi)
        decomposition: The encoder decomposition of the signals
            computed through the encoder.
        x_hat: The reconstction of the input signals, through
            decoder(encoder(x_in[0])).
        auxiliary_models: The list of auxiliary models, [classifier]
        auxiliary_optimizers: The list of auxiliary model optimizers.

    Returns:
        prediction_error: The auxiliary prediction error loss.
        train_enc_dec: Always true, as for this case we do not
            have an uneven update rule.
    """

    _, midi = x_in

    classifier = auxiliary_models[0]
    classifier_optimizer = auxiliary_optimizers[0]

    # Windows overlap by 50%. Hence we only have the MIDI data for
    # every second signal decomposition.
    classifier_input = decomposition[:, ::2, :]
    classifier_input = tf.reshape(classifier_input, (-1, NUMBER_OF_FILTERS))

    with tf.GradientTape() as classifier_tape:
        logits, _ = classifier(classifier_input)
        midi = tf.cast(tf.reshape(midi, (-1, N_MIDI_KEYS)), tf.float32)
        prediction_error = tf.nn.sigmoid_cross_entropy_with_logits(midi, logits)
        prediction_error = tf.reduce_mean(tf.reduce_sum(prediction_error, axis=[1]))

    gradients_of_classifier = classifier_tape.gradient(
        prediction_error, classifier.trainable_variables
    )
    classifier_optimizer.apply_gradients(
        zip(gradients_of_classifier, classifier.trainable_variables)
    )

    return prediction_error, True

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    raw_maestro = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_PATH)
    raw_maestro_midi = maestro_dataset.get_maestro_waveform_dataset(MAESTRO_MIDI_PATH)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    classifier_optimizer = tf.keras.optimizers.Adam(1e-4)

    encoder = learned_basis_function.Encoder(FILTER_LENGTH, NUMBER_OF_FILTERS)
    decoder = learned_basis_function.Decoder(FILTER_LENGTH)
    classifier = learned_basis_function.Classifier(N_MIDI_KEYS)

    learned_decomposition_model = learned_basis_decomposition.LearnedBasisDecomposition(
        encoder, decoder, optimizer, (raw_maestro, raw_maestro_midi), BATCH_SIZE, EPOCHS,
        CHECKPOINT_DIR, RESULTS_DIR, compute_auxiliary_loss_fn=compute_auxiliary_loss,
        auxiliary_models=[classifier], auxiliary_optimizers=[classifier_optimizer]
    )

    learned_decomposition_model.train()

if __name__ == '__main__':
    main()
