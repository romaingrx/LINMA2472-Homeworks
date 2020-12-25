import os

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(root, batch_size, img_size, augmentation=False, cache=False):
    if cache and augmentation:
        raise Exception("Impossible to load augmented data in cache for the moment")
    
    if cache:
        return load_in_cache(root, batch_size, img_size)

    default_augmenation_kwargs = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="constant",
        cval=255
    )

    _aug_kwargs = default_augmenation_kwargs if augmentation else {}

    datagen = ImageDataGenerator(
        rescale=1./255,
        **_aug_kwargs
    )

    dataset = datagen.flow_from_directory(
        root,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None
    )

    return dataset


def load_raw_data(root, img_size, prefix):
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    def _load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = (img - 127.5) / 127.5 # tanh
        return img

    pattern = os.path.join(root, prefix)
    ds = Dataset.list_files(pattern)
    ds = ds.map(_load_img)
    print("FINISHED LOADING THE DATASET")
    return ds

def load_in_cache(root, batch_size, img_size, prefix="*/*"):
    ds = load_raw_data(root, img_size, prefix)
    if img_size[0] > 256:
        ds = ds.cache()
    ds = (ds
          .batch(batch_size)
          .prefetch(-1)
          )
    print("Finished loading dataset")
    return ds

if __name__ == '__main__':
    root_dataset = os.path.join(os.curdir, "bitmoji")
    l = load_in_cache(root_dataset, 64, (64, 64))
    for f in l.take(5):
        print(f.shape)
