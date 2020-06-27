import tensorflow as tf
tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from structures.WaveGAN import Generator, Discriminator
from datasets.NSynthDataset import NSynthTFRecordDataset
import time
import soundfile as sf
import numpy as np
import os
import sys

# Setup Paramaters
D_updates_per_g = 5
Z_dim = 32
BATCH_SIZE = 16
EPOCHS = 100
SEED = tf.random.uniform([5, Z_dim], minval=-1, maxval=1)



# Load the dataset
with tf.device('cpu:0'):
    nsynth_path = "../data/nsynth-train.tfrecord"
    nsynth = NSynthTFRecordDataset(nsynth_path).provide_dataset()
    nsynth = nsynth.batch(BATCH_SIZE).repeat(D_updates_per_g)
    
    
# Construct generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

checkpoint_dir = 'results/WaveGAN/training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def train_step(X, train_generator=True, train_discriminator=True):
    X = tf.reshape(X, shape=(-1, 2**16, 1))
    Z = tf.random.uniform(shape=(X.shape[0], Z_dim), minval=-1, maxval=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        X_gen = generator(Z, training=True)
        X_gen = tf.reshape(X_gen, shape=(-1, 2**16, 1))

        # Compute Wasserstein Distance 
        D_real = discriminator(X, training=True)
        D_fake = discriminator(X_gen, training=True)
        wasserstein_distance = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)

        # Compute gradient penalty
        alpha = tf.random.uniform([X.shape[0], 1, 1], 0.0, 1.0)
        diff = X_gen - X
        interp = X + (alpha * diff)
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

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return G_loss, D_loss

def generate_and_save_audio(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    generations = model(test_input, training=False)
    generations = np.reshape(generations, (-1))

    sf.write('results/WaveGAN/audio/' + str(epoch) + '.wav', generations, 16000)
  

def train():
    print("Training Starting...")
    #generate_and_save_audio(generator, 0, SEED)
    for epoch in range(EPOCHS):
        start = time.time()
        print('Epoch ', epoch, ' starting')

        i = 1
        for X in nsynth:
            G_loss, D_loss = train_step(X, train_generator=False, train_discriminator=True)
            if i % 5 == 0:
                G_loss, D_loss = train_step(X, train_generator=True, train_discriminator=False)
            
            #if i % 500 == 0:
            print("\t[", (time.time()-start) / 60, "] Batch: ", i)
            
            i += 1
            
                

        # Save the model every 15 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('\nTime for epoch {} is {} minutes\n'.format(epoch + 1, (time.time()-start) / 60))
        generate_and_save_audio(generator, epoch+1, SEED)

if __name__ == '__main__':
    train()