from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt

def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def save_imgs(epoch, generator, noise, root="images", name="bitmoji", nrows=5, ncols=5):
    imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(nrows*ncols):
        mapped_img = (imgs[i].numpy()+1)/2
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(mapped_img)
        plt.axis('off')

    fig.savefig(join(root, f"{name}_{epoch}.png"))
