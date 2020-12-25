from os.path import join
import io

import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt


class AttrDict(dict):
    """Like dict but with attribute access and setting"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_true*y_pred, axis=-1)

def discriminator_loss(loss_object, real_output, fake_output, global_batch_size=None):
    real_loss = loss_object(.9 * tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    # if global_batch_size:
    #    return tf.nn.compute_average_loss(total_loss, global_batch_size=global_batch_size)
    return total_loss


def generator_loss(loss_object, fake_output, global_batch_size=None):
    loss = loss_object(tf.ones_like(fake_output), fake_output)
    # if global_batch_size:
    #    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
    return loss


def save_imgs(epoch, generator, noise, root="images", name="bitmoji", nrows=5, ncols=5):
    matplotlib.use('Agg')
    imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(nrows * ncols):
        mapped_img = (imgs[i].numpy() + 1) / 2
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(mapped_img)
        plt.axis('off')

    fig.savefig(join(root, f"{name}_{epoch}.png"))


@tf.function
def normalize_images(images):
    return (images + 1) / 2


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def samples_grid(samples, nrows=8, ncols=8):
    """Return a grid of the samples images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure()
    for i in range(nrows*ncols):
        # Start next subplot.
        plt.subplot(nrows, ncols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = samples[i]
        if x.shape[-1] == 1:
            x = np.reshape(x, [*x.shape[:-1]])
        plt.imshow(x)
    plt.tight_layout(pad=0)
    return figure


def get_best_strategy():
    N_GPUS = -1
    GPU_AVAILABLE = False
    devices = tf.config.list_physical_devices('GPU')
    if devices:
        GPU_AVAILABLE = True
        N_GPUS = len(devices)
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)

    if not GPU_AVAILABLE:
        print("Strategy : One device on CPU (slow)")
        # strategy = tf.distribute.OneDeviceStrategy("CPU")
        strategy = tf.distribute.get_strategy()
    elif N_GPUS == 1:
        print("Strategy : One device on GPU")
        strategy = tf.distribute.OneDeviceStrategy("GPU")
    else:
        print(f"Strategy : Mirrored strategy on {N_GPUS} GPU's ")
        strategy = tf.distribute.MirroredStrategy()

    return strategy
