import tensorflow as tf
from structures.SpecGAN import Generator, Discriminator
from datasets.MAESTRODataset import get_maestro_magnitude_phase_dataset
from tensorflow.keras.utils import Progbar
import time
import soundfile as sf
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
raw_maestro = raw_maestro[:, :, 0:256,:]
print(raw_maestro.shape)
maestro = tf.data.Dataset.from_tensor_slices(raw_maestro).shuffle(2000).repeat(D_updates_per_g).batch(BATCH_SIZE)

# Construct generator and discriminator
generator = Generator(channels=2)
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
    X = tf.reshape(X, shape=(-1, 128, 256, 2))
    Z = tf.random.uniform(shape=(X.shape[0], Z_dim), minval=-1, maxval=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        X_gen = generator(Z, training=True)
        X_gen = tf.reshape(X_gen, shape=(-1, 128, 256, 2))

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

def save_audio(data, name):
    data = np.pad(data, [[0,0],[0,0],[0,1],[0,0]])
    print(data.shape)
    
    waveforms = []
    log_magnitude = data[:,:,:,0]
    ins_freq = data[:,:,:,1]
    
    ins_freq = np.reshape(ins_freq, (-1, 128, 257))
    log_magnitude = np.reshape(log_magnitude, (-1, 128, 257))
    
    magnitude = np.exp(log_magnitude)
    angle = np.array(ins_freq).cumsum(axis=-1)
    phase = (angle + np.pi) % (2 * np.pi) - np.pi
    
    real = magnitude * np.cos(phase)
    img = magnitude * np.sin(phase)

    stft = real + 1j * img
    
    waveforms = tf.signal.inverse_stft(stft, frame_length=512, frame_step=128)
    waveforms = np.reshape(waveforms, (-1))

    sf.write('_results/representation_study/SpecPhaseGAN/audio/' + name + '.wav', waveforms, 16000)

def generate_and_save_audio(model, epoch, test_input):
    generations = model(test_input, training=False)
    save_audio(generations, 'gen_' + str(epoch))

  

def train():
    print("Training Starting...")
    generate_and_save_audio(generator, 0, SEED)
    save_audio(raw_maestro[0:5], 'real_' + str(0))
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