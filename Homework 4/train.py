import os
from tqdm import tqdm
from time import time
from os.path import join

import tensorflow as tf
import tensorboard as tb

from data import load_data
from utils import discriminator_loss, generator_loss, save_imgs
from model import Discriminator, Generator

#from GAN import Discriminator, Generator


def train():



    root_dataset = join(os.curdir, "BitmojiDataset", "images")
    root_images = join(os.curdir, "cached_images")
    img_size = (28, 28)
    channels = 3

    # settting hyperparameter
    latent_dim = 100
    epochs = 800
    batch_size = 256
    buffer_size = 6000
    save_interval = 5


    generator = Generator(channels=channels)
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    #data, info = tfds.load("mnist", with_info=True, data_dir='~/tensorflow_datasets')
    #train_data = data['train']
    #train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)
    train_dataset = load_data(root_dataset, batch_size, img_size, augmentation=False, cache=True)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    if not os.path.exists(root_images):
        os.makedirs(root_images)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise)

            real_output = discriminator(images)
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([batch_size, latent_dim])

    for epoch in range(epochs):
        total_gen_loss = 0
        total_disc_loss = 0

        tgen = tqdm(enumerate(train_dataset), desc=f"Epoch {epoch+1}/{epochs} ")
        for idx, images in tgen:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

            tgen.set_postfix(
                dict(
                    gen_loss=total_gen_loss.numpy()/(idx+1),
                    disc_loss=total_disc_loss.numpy()/(idx+1)
                )
            )

        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed, root=root_images, name="bitmoji")

if __name__ == '__main__':
    train()