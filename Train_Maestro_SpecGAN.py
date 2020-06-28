import tensorflow as tf
from librosa.core import griffinlim
from structures.SpecGAN import Generator, Discriminator
from datasets.MAESTRODataset import get_maestro_magnitude_phase_dataset
from tensorflow.keras.utils import Progbar
import time
import soundfile as sf
import numpy as np
import os
import sys

"""
  Griffin-Lim phase estimation from magnitude spectrum
"""
def invert_spectra_griffin_lim(X_mag, nfft, nhop, ngl):
    X = tf.complex(X_mag, tf.zeros_like(X_mag))

    def b(i, X_best):
        x = tf.signal.inverse_stft(X_best, nfft, nhop)
        X_est = tf.signal.stft(x, nfft, nhop)
        phase = X_est / tf.cast(tf.maximum(1e-8, tf.abs(X_est)), tf.complex64)
        X_best = X * phase
        return i + 1, X_best

    i = tf.constant(0)
    c = lambda i, _: tf.less(i, ngl)
    _, X = tf.while_loop(c, b, [i, X], back_prop=False)

    x = tf.signal.inverse_stft(X, nfft, nhop)
    #x = x[:, :_SLICE_LEN]

    return x

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 64
BATCH_SIZE = 64
EPOCHS = 3000
SEED = tf.random.uniform([5, Z_dim], minval=-1, maxval=1)

# Setup Dataset
maestro_path = 'data/MAESTRO_6h.npz'
raw_maestro = get_maestro_magnitude_phase_dataset(maestro_path)
raw_maestro = raw_maestro[:, :, 0:256,0]
print(raw_maestro.shape)
maestro = tf.data.Dataset.from_tensor_slices(raw_maestro).shuffle(2000).repeat(D_updates_per_g).batch(BATCH_SIZE)

# Construct generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = '_results/representation_study/SpecGAN/training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train_step(X, train_generator=True, train_discriminator=True):
    X = tf.reshape(X, shape=(-1, 128, 256, 1))
    Z = tf.random.uniform(shape=(X.shape[0], Z_dim), minval=-1, maxval=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        X_gen = generator(Z, training=True)
        X_gen = tf.reshape(X_gen, shape=(-1, 128, 256, 1))

        # Compute Wasserstein Distance 
        D_real = discriminator(X, training=True)
        D_fake = discriminator(X_gen, training=True)

        alpha = tf.random.uniform([X.shape[0], 1, 1, 1], 0.0, 1.0)
        diff = X_gen - X
        interp = X + (alpha * diff)
        
        wasserstein_distance = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
            
        with tf.GradientTape() as t:
            t.watch(interp)
            D_interp = discriminator(interp, training=True)
            
        grad = t.gradient(D_interp, [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[2,1]))
        gp = tf.reduce_mean((slopes - 1.0) ** 2.0)
            
        G_loss = tf.reduce_mean(D_fake)
        D_loss = wasserstein_distance + 10.0 * gp

    gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)

    if train_generator:
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    if train_discriminator:
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return G_loss, D_loss

def save_audio(log_mag_spec, name):
    magnitude = tf.exp(log_mag_spec)
    generations = []
    for m in magnitude:
        m = np.reshape(m, (128, 256))
        m = np.pad(m, [[0,0], [0,1]])
        generations.append(invert_spectra_griffin_lim(m, 512, 128, 16))
    generations = np.array(generations)
    generations = np.reshape(generations, (-1))

    sf.write('_results/representation_study/SpecGAN/audio/' + name + '.wav', generations, 16000)

def generate_and_save_audio(model, epoch, test_input):
    log_magnitude_generations = model(test_input, training=False)
    save_audio(log_magnitude_generations, 'gen_' + str(epoch))
  

def train():
    print("Training Starting...")
    generate_and_save_audio(generator, 0, SEED)
    save_audio(raw_maestro[0:5], 'real_0')
    for epoch in range(EPOCHS):
        pb_i = Progbar(len(raw_maestro))
        start = time.time()

        i = 1
        for X in maestro:
            G_loss, D_loss = train_step(X, train_generator=False, train_discriminator=True)
            if i % 5 == 0:
                G_loss, D_loss = train_step(X, train_generator=True, train_discriminator=False)
                pb_i.add(BATCH_SIZE)
            
            i += 1

        # Save the model every 15 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('\nTime for epoch {} is {} minutes'.format(epoch + 1, (time.time()-start) / 60))
        generate_and_save_audio(generator, epoch+1, SEED)
        save_audio(X[0:5], 'real_' + str(epoch+1))

if __name__ == '__main__':
    train()