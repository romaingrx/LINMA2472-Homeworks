img[src$="center"] {
  display:block;
  margin: 0 auto;
}


WGAN-GP
===

The project is to create bitmojis based on a GAN implementation.

Architecture
---

We based our GAN on a Wasserstein Generative Adversial Network with gradient penalty to avoid as much as possible mode collapse and get the more diversity.

### Generator

<p align="center">
  <img src=./ressources/imgs/generator.jpg />
</p>

### Discriminator


<p align="center">
  <img src=./ressources/imgs/discriminator.jpg />
</p>

Preview of the results
---


<p align="center">
  <img src=./ressources/gif/0-669.gif />
</p>
