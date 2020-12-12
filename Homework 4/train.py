import os
from os.path import join

import tensorflow as tf

import callbacks
from data import load_data, load_in_cache
from model import Generator, Discriminator, GAN

N_GPUS=-1; GPU_AVAILABLE=False

devices = tf.config.list_physical_devices('GPU')
if devices:
    GPU_AVAILABLE=True
    N_GPUS=len(devices)
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

if not GPU_AVAILABLE:
    print("Strategy : One device on CPU (slow)")
    #strategy = tf.distribute.OneDeviceStrategy("CPU")
    strategy = tf.distribute.get_strategy()
elif N_GPUS==1:
    print("Strategy : One device on GPU")
    strategy = tf.distribute.OneDeviceStrategy("GPU")
else:
    print(f"Strategy : Mirrored strategy on {N_GPUS} GPU's ")
    strategy = tf.distribute.MirroredStrategy()


root_dataset = join(os.curdir, "wout")
prefix = "*"
#root_images = join(os.curdir, "all_bitmoji_64")
img_size = (128, 128)

latent_dim = 100
epochs = 5000
batch_size = 128
save_interval = 100_000

base_log_dir = join(os.curdir, "runs")
log_dir = join(base_log_dir, f"run_with_background")

global_batch_size = batch_size * strategy.num_replicas_in_sync # To adapt with strategy

train_data = load_in_cache(root_dataset, global_batch_size, img_size, prefix=prefix)

gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

with strategy.scope():
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def loss(*args, **kwargs):
        return tf.reduce_sum(loss_obj(*args, **kwargs)) * (1./global_batch_size)
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    gan = GAN(generator, discriminator, latent_dim, log_dir)
gan.compile(disc_optimizer, gen_optimizer, loss)


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
            callbacks.GenerateSampleGridCallback(join(log_dir, "samples"), every_n_examples=save_interval),
            callbacks.SaveModelCallback(manager, n=save_interval),
            callbacks.LogMetricsCallback(every_n_examples=save_interval)
        ]
    )
except KeyboardInterrupt:
    manager.save()

print("\nDone training.")
