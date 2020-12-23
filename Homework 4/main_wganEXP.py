import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from data import load_raw_data
import utils

if tf.config.list_physical_devices('GPU'):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, enable=True)

class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class SaveGrid(tf.keras.callbacks.Callback):
    def __init__(self, root, every_epochs, ncols=5, nrows=5):
        super(SaveGrid, self).__init__()
        self.every_epochs = every_epochs
        self.ncols = ncols
        self.nrows = nrows
        self.writer = tf.summary.create_file_writer(root)
        self.log_dir = os.path.join(root, "images")
        os.makedirs(self.log_dir, exist_ok=True)
        self.z = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epochs != 0:
            return
        if self.z is None:
            self.z = tf.random.normal((self.ncols*self.nrows, self.model.latent_dim))

        generated_images = .5 * (self.model.g(self.z) + 1)
        figure = utils.samples_grid(generated_images, nrows=self.nrows, ncols=self.ncols)
        figure.savefig(self.log_dir + f"/samples_grid_{epoch}.png")
        image = utils.plot_to_image(figure)
        with self.writer.as_default():
            tf.summary.image(f"Fake samples", image, step=epoch)



class Generator(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        alpha = .2
        w, h, f = 4, 4, 256

        self.network = tf.keras.models.Sequential([
            layers.Input(shape=(latent_dim,)),

            layers.Dense(w * h * f, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),
            layers.Reshape((w, h, f)),

            layers.Conv2DTranspose(f, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),

            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),

            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),
            
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),

            layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),

            layers.Conv2D(3, (4, 4), 2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha),

            layers.Conv2D(3, (4, 4), 1, padding='same', use_bias=False)
        ])

    def call(self, inputs):
        x = self.network(inputs)
        return tf.tanh(x)


class Discriminator(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        alpha = .2
        rate = .2
        const = ClipConstraint(.01)
        self.network = tf.keras.models.Sequential([

            layers.Conv2D(128, (5, 5), strides=2, padding='same', kernel_constraint=const),
            layers.LeakyReLU(alpha),
            layers.Dropout(rate),

            layers.Conv2D(256, (5, 5), strides=2, padding='same', kernel_constraint=const),
            layers.LeakyReLU(alpha),
            layers.Dropout(rate),
            
            layers.Conv2D(512, (5, 5), strides=2, padding='same', kernel_constraint=const),
            layers.LeakyReLU(alpha),
            layers.Dropout(rate),

            layers.Flatten(),
            #layers.Dropout(rate),

            #layers.Dense(100, kernel_constraint=const),
            #layers.Dropout(rate),

            layers.Dense(1)
        ])

    def call(self, inputs):
        return self.network(inputs)


class WGAN(tf.keras.Model):
    def __init__(self, d, g, critic):
        super(WGAN, self).__init__()
        self.critic = critic
        self.latent_dim = g.latent_dim
        self.d = d
        self.g = g

    def compile(self,
                d_optim,
                g_optim,
                #loss_fn
                ):
        super(WGAN, self).compile()
        self.d_optim = d_optim
        self.g_optim = g_optim
        # self.loss_fn = loss_fn

    def get_penalty(self, real, fake, epsilon):
        mixed_images = fake + epsilon * (real - fake)
        with tf.GradientTape() as tape:
            tape.watch(mixed_images)
            mixed_scores = self.d(mixed_images)


        gradient = tape.gradient(mixed_scores, mixed_images)[0]

        gradient_norm = tf.norm(gradient)
        penalty = tf.math.reduce_mean((gradient_norm - 1)**2)
        return penalty

    def discriminator_step(self, real_images, z, epsilon, c_lambda):
        fake_images = self.g(z)
        penalty = self.get_penalty(real_images, fake_images, epsilon)
        with tf.GradientTape() as tape:
            real_output = self.d(real_images)
            fake_output = self.d(fake_images)
            loss = tf.math.reduce_mean(fake_output) - tf.math.reduce_mean(real_output) + c_lambda*penalty

        grads = tape.gradient(loss, self.d.trainable_variables)
        self.g_optim.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def generator_step(self, z):
        with tf.GradientTape() as tape:
            fake_images = self.g(z)
            try_to_fool = self.d(fake_images)
            loss = -tf.math.reduce_mean(try_to_fool)

        grads = tape.gradient(loss, self.g.trainable_variables)
        self.g_optim.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss

    def train_step(self, real_images):
        batch_size = real_images.shape[0]

        dloss = tf.constant(.0)
        #for _ in range(self.critic):
        z = tf.random.normal((batch_size, self.latent_dim))
        epsilon = tf.random.normal((batch_size, 1, 1, 1))
        _dloss = self.discriminator_step(real_images, z, epsilon, 2.)
        dloss += _dloss

        z = tf.random.normal((batch_size//self.critic, self.latent_dim))
        gloss = self.generator_step(z)

        return dict(gloss=gloss, dloss=dloss / self.critic)


NAME = "EXPERIMENTAL"
BATCH_SIZE = 512
LATENT_DIM = 15
OUT_SIZE = 64
EPOCHS = 2000
CRITIC = 8

latest = tf.train.latest_checkpoint(os.path.join(os.curdir, "runs32"))
print(latest)

#strategy = tf.distribute.get_strategy()
strategy = tf.distribute.MirroredStrategy()

#strategy = tf.distribute.OneDeviceStrategy(tf.device("GPU"))

with strategy.scope():
    d = Discriminator()
    g = Generator(LATENT_DIM)
    wgan = WGAN(d, g, CRITIC)
    wgan.compile(
        tf.keras.optimizers.RMSprop(.0001),
        tf.keras.optimizers.RMSprop(.0001),
    )
    if latest:
        wgan.load_weights(latest)


ds = load_raw_data("./datasets/bitmoji_bg_128/", OUT_SIZE, prefix="*")

ds = (ds
      .cache()
      #.shuffle(140_000)
      .batch(BATCH_SIZE*strategy.num_replicas_in_sync, drop_remainder=True)
      .prefetch(-1)
      )

#ds = strategy.experimental_distribute_dataset(ds)

try:
    wgan.fit(
        ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TensorBoard(f"./runs{OUT_SIZE}/{NAME}", histogram_freq=1),
            SaveGrid(f"./runs/{NAME}", 1, 64),
            tf.keras.callbacks.ModelCheckpoint(f"./runs{OUT_SIZE}/{NAME}")
        ]
    )
except KeyboardInterrupt:
    wgan.save_weights(f"./runs{OUT_SIZE}/KEYBOARDINTERRUPT.h5")

