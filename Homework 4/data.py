import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from multiprocessing import Pool

import cv2
import imageio
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset

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

def load_in_cache(root, batch_size, img_size, prefix="*/*"):
    def _load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = (img - 127.5) / 127.5 # tanh
        return img

    pattern = os.path.join(root, prefix)
    ds = Dataset.list_files(pattern)
    for f in ds.take(5):
        print(f)

    ds = (ds
          .map(_load_img, num_parallel_calls=-1)
          .cache()
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
