import math
import numpy as np
from copy import copy as deepcopy
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import Constraint

class Generator(Sequential):
    min_log2 = 4
    min_filters = 16
    w_init = tf.random_normal_initializer(mean=0., stddev=.02)
    gamma_init = tf.random_normal_initializer(mean=1., stddev=.02)
    def __init__(self, latent_dim, out_size, channels, alpha=.2, filters=None, _layers=None, *args, **kwargs):
        self.log2 = math.log2(out_size)
        assert self.log2.is_integer() and out_size >= 8, "Only accept output size which are power of 2 with minimum of 8"
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_size = out_size
        self.channels = channels
        self.alpha = alpha
        if _layers: # Double output size
            self.add(layers.Input((self.latent_dim, )))
            for layer in _layers:
                self.add(layer)
        else: # Build for the first time
            self.dim = 4
            self.first_log2 = Generator.min_log2 + self.log2 - 1
            self.filters = filters or int(2**self.first_log2)
            self.first_filters = self.filters

            self.add(layers.Input((self.latent_dim, )))

            self.add(
                Generator.dense_block((4, 4, self.filters), Generator.w_init, Generator.gamma_init, alpha=.2)
            )
            assert self.output_shape == (None, self.dim, self.dim, self.filters)
            self.add(
                Generator.conv_transpose_block(self.filters, (5, 5), (1, 1), padding="same", w_init=Generator.w_init, gamma_init=Generator.gamma_init, alpha=self.alpha)
            )
            assert self.output_shape == (None, self.dim, self.dim, self.filters)

            while self.dim < self.out_size:
                self.filters //= 2
                self.dim *= 2
                self.add(
                    Generator.conv_transpose_block(self.filters, (5, 5), (2, 2), padding="same", w_init=Generator.w_init, gamma_init=Generator.gamma_init, alpha=self.alpha)
                )
                assert self.output_shape == (None, self.dim, self.dim, self.filters)

            self.add(
                Generator.output_layer(self.channels, (5, 5), padding="same", w_init=Generator.w_init, name=f"Output_{self.dim}")
            )
            assert self.output_shape == (None, self.dim, self.dim, self.channels)

    def double_output_size(self):
        img_size, channels = self._get_output_shape()
        filters = self._get_last_filters()
        new_layers = self.layers[:-1]
        new_layers.append(
            Generator.conv_transpose_block(filters//2, (5, 5), (2, 2), alpha=self.alpha, gamma_init=Generator.gamma_init, padding="same", w_init=Generator.w_init)
        )
        new_layers.append(
            Generator.output_layer(channels, (5, 5), padding="same", w_init=Generator.w_init, name=f"Output_{img_size*2}")
        )
        return Generator(self.latent_dim, img_size*2, channels, alpha=self.alpha, _layers=new_layers)

    def _get_output_shape(self):
        return self.layers[-1].output.shape[2:] # img_size, channels

    def _get_last_filters(self):
        return self.layers[-2].layers[0].filters

    def _turn_last_trainable(self, trainable):
        idx = 0
        for layer in self.layers[::-1]:
            idx += 1
            if idx > 2:
                layer.trainable = trainable
                for l in layer.layers:
                    l.trainable = trainable

    def freeze(self):
        print("FREEZE generator")
        self._turn_last_trainable(False)

    def unfreeze(self):
        print("UNFREEZE generator")
        self._turn_last_trainable(True)

    @classmethod
    def dense_block(cls, reshape_size, w_init, gamma_init, alpha, name=None):
        w, h, f = reshape_size
        name = name or f"dense_block_{f}"
        block_layers = [
            layers.Dense(w * h * f, use_bias=False, kernel_initializer=w_init),
            layers.BatchNormalization(gamma_initializer=gamma_init),
            layers.LeakyReLU(alpha),
            layers.Reshape(reshape_size)
        ]
        return Sequential(block_layers, name=name)

    @classmethod
    def conv_transpose_block(cls, filters, kernel_size, strides, padding, w_init, gamma_init, alpha, name=None):
        #name = name or f"conv_transpose_block_{filters}"
        block_layers = [
            layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=w_init),
            layers.BatchNormalization(gamma_initializer=gamma_init, fused=True),
            layers.LeakyReLU(alpha)
        ]
        return Sequential(block_layers, name=name)

    @classmethod
    def output_layer(cls, filters, kernel_size, padding, w_init, name=None):
        block_layers = [
            layers.Conv2D(filters, kernel_size, padding=padding, use_bias=False, activation='tanh', kernel_initializer=w_init)
        ]
        return Sequential(block_layers, name=name)


class Discriminator(Sequential):
    min_log2 = 1
    min_filters = 16
    base_layers = 3
    w_init = tf.random_normal_initializer(mean=0., stddev=.02)
    gamma_init = tf.random_normal_initializer(mean=1., stddev=.02)
    def __init__(self, in_size, channels, alpha=.2, dropout_rate=.3, filters=None, _layers=None, *args, **kwargs):
        self.log2 = math.log2(in_size)
        assert self.log2.is_integer() and in_size >= 8, "Only accept output size which are power of 2 with minimum of 8"
        super(Discriminator, self).__init__()
        self.in_size = in_size
        self.channels = channels
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        if _layers: # Double input size
            self.add(layers.Input(shape=(self.in_size, self.in_size, self.channels)))
            for layer in _layers:
                self.add(layer)
        else: # Build for the first time
            self.first_log2 = self.min_log2 + self.log2 - 1
            filters = filters or int(2**self.first_log2)

            self.add(layers.Input(shape=(self.in_size, self.in_size, self.channels)))

            for _ in range(self.base_layers):
                self.add(
                    self.conv_block(filters, (5, 5), (2, 2), padding="same", w_init=self.w_init, alpha=self.alpha, dropout_rate=self.dropout_rate)
                )
                filters = filters*2
                
            self.add(
                self.output_block(Discriminator.w_init)
            )

    def _turn_last_trainable(self, trainable):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "layers"):
                idx += 1
                if idx > 2:
                    layer.trainable = trainable
                    for l in layer.layers:
                        l.trainable = trainable

    def freeze(self):
        print("FREEZE discriminator")
        self._turn_last_trainable(False)

    def unfreeze(self):
        print("UNFREEZE discriminator")
        self._turn_last_trainable(True)

    def double_input_size(self):
        channels = self.channels; img_size = self.in_size
        filters = self._get_last_filters()

        keeped_layers = self.layers[1:]
        new_input = layers.Input(shape=(img_size*2, img_size*2, channels))
        new_conv = Discriminator.conv_block(filters//2, (5, 5), (2, 2), padding="same", w_init=Discriminator.w_init,
                                            alpha=self.alpha, dropout_rate=self.dropout_rate)
        new_conv_replace = Discriminator.conv_block(filters, (5, 5), (2, 2), padding="same", w_init=Discriminator.w_init,
                                                alpha=self.alpha, dropout_rate=self.dropout_rate)
        new_layers = [
            #new_input,
            new_conv,
            new_conv_replace,
            *keeped_layers
        ]
        return Discriminator(img_size*2, channels, alpha=self.alpha, dropout_rate=self.dropout_rate, _layers=new_layers)

    def _get_last_filters(self):
        for l in self.layers:
            if hasattr(l, "layers"):
                return l.layers[0].filters

    @classmethod
    def conv_block(cls, filters, kernel_size, strides, padding, w_init, alpha, dropout_rate, name=None):
        #name = name or f"conv_block_{filters}"
        block_layers = [
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=w_init),
            layers.LeakyReLU(alpha),
            layers.Dropout(dropout_rate)

        ]
        return Sequential(block_layers, name=name)

    @classmethod
    def output_block(cls, w_init, name=None):
        name = name or "output_classifier"
        block_layers = [
            layers.Flatten(),
            layers.Dense(1, kernel_initializer=w_init, activation="sigmoid"),
        ]
        return Sequential(block_layers, name=name)

    def _get_weights(self, layers=None):
        layers = layers or self.layers
        ret_dict = {}
        for l in layers:
            ret_dict[l.name] = self._get_weights(l.layers) if hasattr(l, "layers") else l.get_weights()
        return ret_dict

    def _set_weights(self, weights, layers=None):
        layers = layers or self.layers
        for l in layers:
            if hasattr(l, "layers"):
                self._set_weights(weights[l.name], layers=l.layers)
            else:
                l.set_weights(weights[l.name])
        return self


class GAN(tf.keras.Model):

    def __init__(self, generator, discriminator, latent_dim, log_dir):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.latent_dim = latent_dim

        self.summary_writer = tf.summary.create_file_writer(log_dir)

        self.n_img = tf.Variable(0, dtype=tf.int64, trainable=False, name="n_img")
        #self.n_batches = tf.Variable(0, dtype=tf.int64, trainable=False, name="n_batches")

        self.batch_size = None

        #self.strategy = tf.distribute.get_strategy()

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn


    def D_step(self, real_images, Z=None):
        batch_size = tf.shape(real_images)[0]
        if Z is None:
            Z = tf.random.normal((batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(Z)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), .9*tf.ones((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        #labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        return d_loss

    def G_step(self, Z, batch_size):
        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(Z))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return g_loss

    def train_step(self, real_images):
        tf.summary.experimental.set_step(self.n_img)

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        d_loss = self.D_step(real_images)

        # Sample random points in the latent space
        Z = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        g_loss = self.G_step(Z, batch_size)

        return {"d_loss": d_loss, "g_loss": g_loss, "size" : batch_size}

    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.input_shape[-1]])

    def generate_samples(self, latents=None, training=False):
        if latents is None:
            latents = self.latents_batch()
        return self.generator(latents, training=training)

    #def double_size(self):
    #    self.generator = self.generator.double_output_size()
    #    self.discriminator = self.discriminator.double_input_size()

    def summary(self):
        #print("> Generator")
        self.generator.summary()
        #print("> Discriminator")
        self.discriminator.summary()



class UnrolledDCGAN(GAN):
    def __init__(self, generator, discriminator, latent_dim, n_unrolled, log_dir):
        #assert isinstance(tf.distribute.get_strategy(), tf.distribute.OneDeviceStrategy), f"Does not work with multi-Devices strategies :: current strategy {tf.distribute.get_strategy()}"
        super(UnrolledDCGAN, self).__init__(generator, discriminator, latent_dim, log_dir)
        self.n_unrolled = n_unrolled

    def G_step(self, real_image, Z, batch_size):

        if self.n_unrolled > 0:
            D_backup = deepcopy(self.discriminator)
            for _ in range(self.n_unrolled):
                self.D_step(real_image, Z)
        
        g_loss = super().G_step(Z, batch_size)

        if self.n_unrolled > 0:
            self.discriminator = D_backup

        return g_loss

    def train_step(self, real_images):
        tf.summary.experimental.set_step(self.n_img)

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        d_loss = self.D_step(real_images)

        # Sample random points in the latent space
        Z = tf.random.normal(shape=(batch_size, self.latent_dim))

        g_loss = self.G_step(real_images, Z, batch_size)

        return {"d_loss": d_loss, "g_loss": g_loss, "size" : batch_size}


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class WGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(WGenerator, self).__init__(*args, **kwargs)

class WDiscriminator(Discriminator):
    clip_const = ClipConstraint(.01)
    def __init__(self, in_size, channels, alpha=.2, dropout_rate=.3, filters=None, _layers=None, *args, **kwargs):
        super(WDiscriminator, self).__init__(in_size, channels, alpha, dropout_rate, filters, _layers, *args, **kwargs)

    @classmethod
    def conv_block(cls, filters, kernel_size, strides, padding, w_init, alpha, dropout_rate, name=None):
        #name = name or f"conv_block_{filters}"
        block_layers = [
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=w_init, kernel_constraint=WDiscriminator.clip_const),
            layers.LeakyReLU(alpha),
            layers.Dropout(dropout_rate)

        ]
        return Sequential(block_layers, name=name)

    @classmethod
    def output_block(cls, w_init, name=None):
        name = name or "output_classifier"
        block_layers = [
            layers.Flatten(),
            layers.Dense(1, kernel_initializer=w_init, activation="linear"),
        ]
        return Sequential(block_layers, name=name)

    def train_step(self, batch):
        real_images, fake_images = batch
        batch_size = real_images.shape[0]
        y_real = -1 * tf.ones((batch_size, 1))
        y_fake = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            y_hat_real = self.discriminator(real_images)
            y_hat_fake = self.discriminator(fake_images)
            d_loss_real = self.loss(y_real, y_hat_real)
            d_loss_fake = self.loss(y_fake, y_hat_fake)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return dict(d_loss=d_loss, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake)



class WGAN(GAN):
    def __init__(self, generator, discriminator, latent_dim, log_dir, critic_size):
        super(WGAN, self).__init__(generator, discriminator, latent_dim, log_dir)
        self.critic_size = critic_size
        #self.batch_size = next(iter(dataset)).shape[0]
        #self.dataset = np.array(list(dataset.unbatch()))
        #self.n_critic = n_critic

    @classmethod
    def loss(cls):
        @tf.function
        def wassertein_loss(y_true, y_pred):
            return tf.reduce_mean(y_true*y_pred)
        return wassertein_loss

    def Dstep(self, X, y):
        with tf.GradientTape() as tape:
            y_hat = self.discriminator(X)
            loss = self.loss_fn(y, y_hat)

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return loss

    def Gstep(self, Z):
        with tf.GradientTape() as tape:
            y_hat = self.discriminator(self.generator(Z))
            loss = - tf.math.reduce_mean(y_hat)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss
    
    def gradient_penalty(self, real, fake, epsilon): 
        #mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_images = fake + epsilon * (real - fake)
        with tf.GradientTape() as tape:
            tape.watch(mixed_images) 
            mixed_scores = self.discriminator(mixed_images)


        gradient = tape.gradient(mixed_scores, mixed_images)[0]
    
        gradient_norm = tf.norm(gradient)
        penalty = tf.math.reduce_mean((gradient_norm - 1)**2)
        return penalty

    def D_critic_step(self, real_images, Z):
        with tf.GradientTape() as tape:
            fake_images = self.generator(Z)
            loss = tf.math.reduce_mean(self.discriminator(fake_images)) - tf.math.reduce_mean(self.discriminator(real_images))

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return loss

    def train_step(self, batch_images):
        batch_size = tf.shape(batch_images)[0]
        self.batch_size = batch_size

        d_loss_track = tf.constant(.0)
        for _ in range(self.critic_size):
            noise = tf.random.normal((batch_size, self.latent_dim))
            d_loss = self.D_critic_step(batch_images, noise)
            d_loss_track += d_loss

        g_loss = self.Gstep(noise)
        d_loss_track /= self.critic_size

        return dict(g_loss=g_loss, d_loss=d_loss_track, size=self.batch_size)

    def dum_train_step(self, batch_images):
        batch_size = tf.shape(batch_images)[0]
        self.batch_size = batch_size
        valid = -1*tf.ones((batch_size, 1))
        fake = 1*tf.ones((batch_size, 1))
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        for _ in range(self.critic_size):               
            
            with tf.GradientTape() as dtape:
                fake_images = self.generator(noise)
                real_output = self.discriminator(batch_images)
                fake_output = self.discriminator(fake_images)

                epsilon = tf.random.normal((self.batch_size, 1, 1, 1), .0, 1.)
                gp = self.gradient_penalty(batch_images, fake_images, epsilon)

                c_lambda = 10
                d_loss = tf.math.reduce_mean(fake_output) - tf.math.reduce_mean(real_output) + c_lambda*gp
        
            d_gradients = dtape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise)
            fake_output = self.discriminator(fake_images)
            g_loss = -1.*tf.math.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))


        return dict(g_loss=g_loss, d_loss=d_loss, size=self.batch_size)
        #return dict(g_loss=g_loss, d_loss=d_loss, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake, size=self.batch_size)

    def _train_step(self, batch_images):
        assert len(batch_images.shape) == 5, "Need 5 dims tensor to train :: shape (BATCH_SIZE,N_CRITIC,H,W,C)"
        batch_size = tf.shape(batch_images)[0]

        valid = -tf.ones((batch_size, 1))
        fake = tf.ones((batch_size, 1))

        critic_batch_images = tf.transpose(batch_images, [1, 0, 2, 3, 4])
        x = [0]*self.critic_size
        for i in range(self.critic_size):
            Z = tf.random.normal((batch_size, self.latent_dim))
            x[i] = self.generator(Z)
        critic_batch_fake_images = tf.transpose(tf.convert_to_tensor(x), [1,0,2,3,4])
        Ddata = tf.data.Dataset.from_tensor_slices((critic_batch_images, critic_batch_fake_images)).cache()
        for real_images, fake_images in Ddata:
            d_loss_real = self.Dstep(real_images, valid)
            d_loss_fake = self.Dstep(fake_images, fake)
            d_loss = .5 * (d_loss_real + d_loss_fake)

        #hist = self.discriminator.fit(Ddata, epochs=1, verbose=2).history
        #d_loss = tf.reduce_mean(hist["d_loss"])
        #d_loss_real = tf.reduce_mean(hist["d_loss_real"])
        #d_loss_fake = tf.reduce_mean(hist["d_loss_fake"])

        #for valid_images in critic_batch_images:
        #    Z = tf.random.normal((batch_size, self.latent_dim))
        #    fake_images = self.generator(Z)
        #    d_loss_real = self.Dstep(valid_images, valid)
        #    d_loss_fake = self.Dstep(fake_images, fake)
        #    d_loss = .5 * (d_loss_real + d_loss_fake)

        g_loss = self.Gstep(Z, valid)

        return dict(g_loss=g_loss, d_loss=d_loss, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake, size=self.batch_size)
        
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile(d_optimizer, g_optimizer, loss_fn)
        self.generator.compile(optimizer=self.g_optimizer, loss=self.loss_fn)
        self.discriminator.compile(optimizer=self.d_optimizer, loss=self.loss_fn)


if __name__ == '__main__':
    data = tf.data.Dataset.from_tensor_slices(tf.random.normal((1000, 64, 64, 3))).batch(32).unbatch()
    l = list(data)
    print(type(l[0]))
