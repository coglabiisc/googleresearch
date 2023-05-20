"""This module contains some utility functions for loading data."""

import csv
import functools
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

partial = functools.partial
    
def constant_generator(split=b'val', mode=b'grayscale'):
  if mode == b'grayscale':
    images = np.ones((256,32,32,1))
    for i in range (images.shape[0]):
      images[i,:,:,:] *= i
    
    for image in images:
      yield image
  else:
    np.random.seed(0)
    images = np.ones((10000,32,32,3))
    for i in range (images.shape[0]):
      for j in range (images.shape[-1]):
        images[i,:,:,j] *= np.random.randint(0, high=256)
    
    for image in images:
      yield image
  

def noise_generator(split=b'val', mode=b'grayscale'):
  """Generator function for the noise dataest.

  Args:
    split: Data split to load - "train", "val" or "test"
    mode: Load in "grayscale" or "color" modes
  Yields:
    An noise image
  """
  if split == b'train':
    np.random.seed(0)
  if split == b'val':
    np.random.seed(1)
  else:
    np.random.seed(2)
  for _ in range(10000):
    if mode == b'grayscale':
      yield np.random.randint(low=0, high=256, size=(32, 32, 1))
    else:
      yield np.random.randint(low=0, high=256, size=(32, 32, 3))


def compcars_generator(split=b'train'):
  """Generator function for the CompCars Surveillance dataest.

  Source: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

  Args:
    split: Data split to load - "train", "val" or "test".

  Yields:
    An image
  """

  rootpath = '../data/compcars/sv_data'
  random.seed(42)
  # random.seed(43)  # split 2

  if split in [b'train', b'val']:
    split_path = os.path.join(rootpath, 'train_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))

  elif split == b'test':
    split_path = os.path.join(rootpath, 'test_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))
    
def clevr_generator(split=b'train'):
  """Generator function for the CompCars Surveillance dataest.

  Source: https://cs.stanford.edu/people/jcjohns/clevr/

  Args:
    split: Data split to load - "train", "val" or "test".

  Yields:
    An image
  """
    
  rootpath = '../data/clevr/CLEVR_v1.0/'
  random.seed(42)
  
  if split in [b'train', b'val']:
    all_images = os.listdir(rootpath + 'images/train/')
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath + 'images/train/' + image_name))[:,:,:3]*255
    
  elif split == b'test':
    all_images = os.listdir(rootpath + 'images/test/')
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath + 'images/test/' + image_name))[:,:,:3]*255
    
def gtsrb_generator(split=b'train', cropped=False):
  """Generator function for the GTSRB Dataset.

  Source: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

  Args:
    split: Data split to load - "train", "val" or "test".
    cropped: Whether to load cropped version of the dataset.

  Yields:
    An image
  """

  rootpath = '../data/gtsrb/GTSRB'
  random.seed(42)
  # random.seed(43)  # split 2
  if split in [b'train', b'val']:
    rootpath = os.path.join(rootpath, 'Final_Training', 'Images')
    all_images = []
    # loop over all 42 classes
    for c in range(0, 43):
      # subdirectory for class
      prefix = rootpath + '/' + format(c, '05d') + '/'
      # annotations file
      gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
      # csv parser for annotations file
      gt_reader = csv.reader(gt_file, delimiter=';')
      next(gt_reader)  # skip header

      for row in gt_reader:
        all_images.append((prefix + row[0],
                           (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
                          ))
      gt_file.close()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image, bbox in all_images:
      img = plt.imread(image)
      if cropped:
        img = img[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1, :]
      yield img

  elif split == b'test':
    rootpath = os.path.join(rootpath, 'Final_Test', 'Images/')
    gt_file = open(rootpath + '/GT-final_test.test.csv')
    gt_reader = csv.reader(gt_file, delimiter=';')
    next(gt_reader)
    for row in gt_reader:
      img = plt.imread(rootpath + row[0])
      if cropped:
        bbox = (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
        img = img[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1, :]
      yield img
    gt_file.close()

def celeba_generator(split=b'train'):
  """Generator function for the GTSRB Dataset.

  Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

  Args:
    split: Data split to load - "train", "val" or "test".
    cropped: Whether to load cropped version of the dataset.

  Yields:
    An image
  """

  rootpath = '../data/celeb_a/images/'
  random.seed(42)
  all_images = [f for f in os.listdir(rootpath)]
  
  if split in [b'train', b'val']:
      all_images = all_images[:162770]
      random.shuffle(all_images)
      if split == b'train':
          all_images = all_images[:-(len(all_images)//10)]
      else:
          all_images = all_images[-(len(all_images)//10):]
      for image_name in all_images:
          yield plt.imread(os.path.join(rootpath, image_name))

  elif split == b'test':
      all_images = all_images[182637:]
      for image_name in all_images:
          yield plt.imread(os.path.join(rootpath, image_name))

def sign_lang_generator(split=b'val'):

  """Generator function for the GTSRB Dataset.

  Source: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

  Args:
    split: Data split to load - "train", "val" or "test".
    cropped: Whether to load cropped version of the dataset.

  Yields:
    An image
  """
    
  rootpath = '../data/sign_lang/Hand_sign_mnist/'
  random.seed(42)
  
  if split in [b'train', b'val']:
      all_images = []
      for root, dirs, files in os.walk(rootpath + 'Train'):
            for file in files:
              all_images.append(os.path.join(root, file))
      random.shuffle(all_images)
      if split == b'train':
          all_images = all_images[:-(len(all_images)//10)]
      else:
          all_images = all_images[-(len(all_images)//10):]
      for image_name in all_images:
          yield np.expand_dims(plt.imread(os.path.join(image_name)),-1)

  elif split == b'test':
      all_images = []
      for root, dirs, files in os.walk(rootpath + 'Test'):
            for file in files:
              all_images.append(os.path.join(root, file))
      for image_name in all_images:
          yield np.expand_dims(plt.imread(os.path.join(image_name)),-1)

def cifar10_class_generator(split, cls):
  """Generator function of class wise CIFAR10 dataset.

  Args:
    split: Data split to load - "train", "val" or "test".
    cls: The target class to load examples from.

  Yields:
    An image
  """
  (ds_train, ds_val, ds_test) = tfds.load('cifar10',
                                          split=['train[:90%]',
                                                 'train[90%:]',
                                                 'test'
                                                 ],
                                          as_supervised=True)

  if split == b'train':
    ds = ds_train
  elif split == b'val':
    ds = ds_val
  else:
    ds = ds_test

  for x, y in ds:
    if y == cls:
      yield x
    
def mnist_class_generator(split, cls):
  """Generator function of class wise MNIST dataset.

  Args:
    split: Data split to load - "train", "val" or "test".
    cls: The target class to load examples from.

  Yields:
    An image
  """
  (ds_train, ds_val, ds_test) = tfds.load('mnist',
                                          split=['train[:90%]',
                                                 'train[90%:]',
                                                 'test'
                                                 ],
                                          as_supervised=True)

  if split == b'train':
    ds = ds_train
  elif split == b'val':
    ds = ds_val
  else:
    ds = ds_test

  for x, y in ds:
    if y == cls:
      yield x

def get_dataset(name,
                batch_size,
                mode,
                shuffle_train=True,
                mutation_rate=None
                ):
  """Returns the required dataset with custom pre-processing.

  Args:
    name: Name of the dataset. Supported names are:
      svhn_cropped
      cifar10
      celeb_a
      gtsrb
      compcars
      lsun
      mnist
      fashion_mnist
      emnist_letters
      sign_lang
      clevr
      noise
      constant
    batch_size: Batch Size
    mode: Load in "grayscale" or "color" modes
    normalize: Type of normalization to apply. Supported values are:
      None
      pctile-x (x is an integer)
      histeq
    dequantize: Whether to apply uniform dequantization
    shuffle_train: Whether to shuffle examples in the train split
    visible_dist: Visible dist of the model

  Returns:
    The train, val and test splits respectively
  """

  def preprocess(image, inverted, mode, mutation_rate=None):
    if isinstance(image, dict):
      image = image['image']
    
    image = tf.image.resize(image, [32, 32], antialias=True)
    image = tf.cast(tf.round(image), tf.int32)

    if mode == 'grayscale':
      if image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    else:
      if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)

    if mutation_rate != 0:
        w, h, c = image.get_shape().as_list()
        mask = tf.cast(
              tf.compat.v1.multinomial(
                  tf.compat.v1.log([[1.0 - mutation_rate, mutation_rate]]), w * h * c),
              tf.int32)[0]
        mask = tf.reshape(mask, [w, h, c])
        possible_mutations = tf.compat.v1.random_uniform(
              [w * h * c],
              minval=0,
              maxval=256,
              dtype=tf.int32)
        possible_mutations = tf.reshape(possible_mutations, [w, h, c])
        image = tf.compat.v1.mod(image + mask * possible_mutations, 256)
    image = tf.cast(tf.round(image), tf.float32)

    if inverted:
      image = 255 - image
    else:
      return image

  assert name in ['svhn_cropped', 'cifar10', 'celeb_a', 'gtsrb', 'compcars', 'constant',
                  'mnist', 'fashion_mnist', 'sign_lang', 'emnist_letters', 'noise', 'lsun', 'clevr',
                  *[f'cifar10-{i}' for i in range(10)], *[f'mnist-{i}' for i in range(10)]], \
      f'Dataset {name} not supported'

  if name == 'noise':
    n_channels = 1 if mode == 'grayscale' else 3
    ds_train = tf.data.Dataset.from_generator(
        noise_generator,
        args=['train', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_val = tf.data.Dataset.from_generator(
        noise_generator,
        args=['val', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_test = tf.data.Dataset.from_generator(
        noise_generator,
        args=['test', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    n_examples = 1024
  elif name.startswith('gtsrb'):
    ds_train = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['train', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['val', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['test', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'compcars':
    ds_train = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'clevr':
    ds_train = tf.data.Dataset.from_generator(
        clevr_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        clevr_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        clevr_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'celeb_a':
    ds_train = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'sign_lang':
    ds_train = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_val = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_test = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    n_examples = 1024
  elif name.startswith('cifar10-'):
    n_examples = 1024
    cls = int(name.split('-')[1])
    ds_train = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['train', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['val', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['test', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
  elif name.startswith('mnist-'):
    n_examples = 1024
    cls = int(name.split('-')[1])
    ds_train = tf.data.Dataset.from_generator(
        mnist_class_generator,
        args=['train', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_val = tf.data.Dataset.from_generator(
        mnist_class_generator,
        args=['val', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_test = tf.data.Dataset.from_generator(
        mnist_class_generator,
        args=['test', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
  elif name == 'constant':
    n_channels = 1 if mode == 'grayscale' else 3
    ds_train = tf.data.Dataset.from_generator(
        constant_generator,
        args=['train', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_val = tf.data.Dataset.from_generator(
        constant_generator,
        args=['val', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_test = tf.data.Dataset.from_generator(
        constant_generator,
        args=['test', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    n_examples = 256
  elif name.startswith('lsun'):
    name = name.replace('_', '/', 1)
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        name, split=['train[10%:90%]', 'train[:10%]', 'train[90%:]'], with_info=True)
    n_examples = ds_info.splits['train'].num_examples
  elif name == 'emnist_letters':
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'emnist/letters', split=['train[10%:]', 'train[:10%]', 'test'], with_info=True)
    n_examples = ds_info.splits['train'].num_examples
  else:
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        name, split=['train[10%:]', 'train[:10%]', 'test'], with_info=True)
    n_examples = ds_info.splits['train'].num_examples

  ds_train = ds_train.map(
      partial(preprocess, mode=mode, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.cache()
  if shuffle_train:
    ds_train = ds_train.shuffle(n_examples)
  ds_train = ds_train.batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_val = ds_val.map(
      partial(preprocess, mode=mode, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_val = ds_val.cache()
  ds_val = ds_val.batch(batch_size)
  ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.map(
      partial(preprocess, mode=mode, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_val, ds_test
