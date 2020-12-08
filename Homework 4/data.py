import os
from tqdm import tqdm

import cv2
import imageio
import numpy as np

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


def load_in_cache(root, batch_size, img_size):
    _imgs = []
    for _root, dirs, files in os.walk(root):
        for fname in files:
            if fname.split('.')[-1].lower() in ('jpg', 'png'):
                img = imageio.imread(os.path.join(_root, fname))
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
                img = np.divide(img, 255.)
                _imgs.append(img)

    nd_img = np.stack(_imgs, axis=0)
    dataset = Dataset.from_tensor_slices(nd_img).batch(batch_size).prefetch(-1)

    return dataset

