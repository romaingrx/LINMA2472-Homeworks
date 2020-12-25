WGAN-GP
===

The project is to create bitmojis based on a GAN implementation.

Architecture
---

We based our GAN on a Wasserstein Generative Adversial Network with gradient penalty to avoid as much as possible mode collapse and get the more diversity.

### Generator

![Generator architecture](./ressources/imgs/generator.jpg)

### Discriminator

![Discriminator architecture](./ressources/imgs/discriminator.jpg)

Preview of the results
---

![0-669 epochs generaated images](./ressources/gif/0-669.gif)