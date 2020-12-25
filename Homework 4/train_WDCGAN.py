import os
from os.path import join
import logging

import tensorflow as tf

import callbacks
from utils import get_best_strategy, AttrDict, WassersteinLoss
from data import load_data, load_in_cache, load_raw_data
from models import WGenerator, WDiscriminator, WGAN

root_dataset = join(os.curdir, "datasets", "wout")
prefix = "*"

EPOCHS = 2000

CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 64
N_CRITIC = 5
SIZE = 64
SAVE_INTERVAL = 1099*64

BASE_LOG_DIR = join(os.curdir, "runs")
LOG_DIR = join(BASE_LOG_DIR, f"wgan_test_64")

logger = logging.getLogger("WDCGAN")
logger.setLevel(logging.DEBUG)

gen_optimizer = tf.keras.optimizers.RMSprop(0.00005)
disc_optimizer = tf.keras.optimizers.RMSprop(0.00005)

strategy = get_best_strategy()
#strategy = tf.distribute.get_strategy()
global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync # To adapt with strategy

with strategy.scope():
    loss_obj = WassersteinLoss(reduction=tf.keras.losses.Reduction.NONE)
    def loss(*args, **kwargs):
        return tf.reduce_sum(loss_obj(*args, **kwargs)) * (1./global_batch_size)

if not os.path.exists(LOG_DIR):os.makedirs(LOG_DIR)
img_dir = join(LOG_DIR, f"samples_{SIZE}")
#tf.debugging.experimental.enable_dump_debug_info(inter_logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

train_data = load_raw_data(root_dataset, (SIZE, SIZE), prefix=prefix)
train_data = train_data.batch(global_batch_size).prefetch(-1)

for X in train_data:
    print(f"Sample critic batch shape: {X.shape}")
    break

with strategy.scope():
    generator = WGenerator(LATENT_DIM, SIZE, CHANNELS)
    discriminator = WDiscriminator(SIZE, CHANNELS)
    gan = WGAN(generator, discriminator, LATENT_DIM, LOG_DIR, N_CRITIC)
gan.compile(disc_optimizer, gen_optimizer, loss)

checkpoint = tf.train.Checkpoint(gan=gan)
manager = tf.train.CheckpointManager(
    checkpoint,
    directory=LOG_DIR,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=1
)

gan.summary()

try:
    gan.fit(
        train_data,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TensorBoard(LOG_DIR),
            callbacks.GenerateSampleGridCallback(img_dir, every_n_examples=SAVE_INTERVAL),
            tf.keras.callbacks.ModelCheckpoint(LOG_DIR, save_weights_only=True, verbose=1)
            #callbacks.SaveModelCallback(manager, n=SAVE_INTERVAL),
     ]
    )
except KeyboardInterrupt:
	pass
finally:
    manager.save()

print("\nDone training.")
