import os
from os.path import join
import logging

import tensorflow as tf

import callbacks
from utils import get_best_strategy, AttrDict
from data import load_data, load_in_cache
from model import Generator, Discriminator, GAN

root_dataset = join(os.curdir, "wout")
prefix = "*"

INTER_SIZE = [32, 64, 128, 256]
INTER_EPOCHS = [200, 200, 200, 3400]

LATENT_DIM = 100
BATCH_SIZE = 128
SAVE_INTERVAL = 100_000

BASE_LOG_DIR = join(os.curdir, "runs")
LOG_DIR = join(base_log_dir, f"run_with_background")

logger = logging.getLogger("DCGAN")
logger.setLevel(logging.DEBUG)

gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

strategy = get_best_strategy()
global_batch_size = batch_size * strategy.num_replicas_in_sync # To adapt with strategy

with strategy.scope():
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def loss(*args, **kwargs):
        return tf.reduce_sum(loss_obj(*args, **kwargs)) * (1./global_batch_size)

for epochs, size in zip(INTER_EPOCHS, INTER_SIZE):
    img_size = (size, size)
    img_dir = join(LOG_DIR, f"samples_{size}")

    if "gan" not in locals():
        with strategy.scope():
            generator = Generator(latent_dim, size=size)
            discriminator = Discriminator(size=size)
            gan = GAN(generator, discriminator, latent_dim, log_dir)
        gan.compile(disc_optimizer, gen_optimizer, loss)
        logger.info(f"--- Compiled the gan for the first time with size of generated images to ({size:d}, {size:d}, 3)")
    else:
        gan.augment_dim(size)
        logger.info(f"--- Augmented the size of generated images to ({size:d}, {size:d}, 3)")

    train_data = load_in_cache(root_dataset, global_batch_size, img_size, prefix=prefix)
    logger.info(f"--- Loaded all images with size {size} ---")

    checkpoint = tf.train.Checkpoint(gan=gan)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=log_dir,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=1
    )

    gan.summary()

    try:
        gan.fit(
            train_data,
            epochs=epochs,
            callbacks=[
                callbacks.GenerateSampleGridCallback(img_dir, every_n_examples=save_interval),
                callbacks.SaveModelCallback(manager, n=save_interval),
                callbacks.LogMetricsCallback(every_n_examples=save_interval)
            ]
        )
    except KeyboardInterrupt:
        manager.save()

print("\nDone training.")
