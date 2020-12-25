import os
from os.path import join
import logging

import tensorflow as tf

import callbacks
from utils import get_best_strategy, AttrDict
from data import load_data, load_in_cache
from models import Generator, Discriminator, UnrolledDCGAN, GAN

root_dataset = join(os.curdir, "datasets", "wout")
prefix = "*"

INTER_SIZE = [64, 128, 256]
INTER_EPOCHS = [1000, 2000, 5000]
#INTER_EPOCHS = [1, 1, 1, 1, 1]
FREEZE_FOR_EPOCHS = 50

CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 128
N_UNROLLED = 0
SAVE_INTERVAL = 100_000

BASE_LOG_DIR = join(os.curdir, "runs")
LOG_DIR = join(BASE_LOG_DIR, f"run_more_epochs_64")

logger = logging.getLogger("DCGAN")
logger.setLevel(logging.DEBUG)

gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

#strategy = get_best_strategy()
#strategy = tf.distribute.get_strategy()
strategy = tf.distribute.MirroredStrategy(
    #cross_device_ops=tf.distribute.NcclAllReduce()
    #cross_device_ops=tf.distribute.ReductionToOneDevice("GPU:0")
)
global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync # To adapt with strategy

with strategy.scope():
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def loss(*args, **kwargs):
        return tf.reduce_sum(loss_obj(*args, **kwargs)) * (1./global_batch_size)

for epochs, size in zip(INTER_EPOCHS, INTER_SIZE):
    for elem in ("train_data", "gan", "checkpoint", "manager"):
        if elem in locals():
            del elem
    inter_logdir = join(LOG_DIR, f"run_{size}")
    if not os.path.exists(inter_logdir):os.makedirs(inter_logdir)
    img_size = (size, size)
    img_dir = join(inter_logdir, f"samples_{size}")
    #tf.debugging.experimental.enable_dump_debug_info(inter_logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    inter_callbacks = []

    with strategy.scope():
        if "generator" in locals():
            generator = generator.double_output_size()
            discriminator = discriminator.double_input_size()
            inter_callbacks.append(
                callbacks.FreezeGenDisc(FREEZE_FOR_EPOCHS)
            )
            logger.info(f"--- Augmented the size of generated images to ({size:d}, {size:d}, 3)")
        else :
            generator = Generator(LATENT_DIM, size, CHANNELS)
            discriminator = Discriminator(size, CHANNELS)
            logger.info(f"--- Compiled the gan for the first time with size of generated images to ({size:d}, {size:d}, 3)")
        gan = GAN(generator, discriminator, LATENT_DIM,inter_logdir)
    gan.compile(disc_optimizer, gen_optimizer, loss)

    train_data = load_in_cache(root_dataset, global_batch_size, img_size, prefix=prefix)
    logger.info(f"--- Loaded all images with size {size} ---")

    checkpoint = tf.train.Checkpoint(gan=gan)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=inter_logdir,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=1
    )

    gan.summary()

    try:
        gan.fit(
            train_data,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(inter_logdir, histogram_freq=1),
                callbacks.GenerateSampleGridCallback(img_dir, every_n_examples=SAVE_INTERVAL),
                callbacks.SaveModelCallback(manager, n=SAVE_INTERVAL),
                *inter_callbacks
                #callbacks.LogMetricsCallback(every_n_examples=SAVE_INTERVAL),
    ]
        )
    except KeyboardInterrupt:
    	pass
    finally:
        manager.save()

print("\nDone training.")
