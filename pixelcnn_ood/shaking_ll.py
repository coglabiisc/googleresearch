import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import tqdm
from scipy.io import savemat
from scipy.io import loadmat
import os
from scipy import stats

import sys
sys.path.append('../Dependencies/')
import dataset_utils
import network
from transform_utils import *

tf.random.set_seed(42)

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

train_set = 'mnist'
input_shape = (32,32,3)
n_patches = 2
orders, shuffles = derangement(n_patches)
mode = 'grayscale'

if mode == 'color':
    input_shape = (32, 32, 3)
    datasets = [
        'svhn_cropped',
        'cifar10',
        'celeb_a',
        'gtsrb',
        'compcars',
        'lsun',
        'noise',
        'constant'
    ]
    num_filters = 64
elif mode == 'grayscale':
    input_shape = (32, 32, 1)
    datasets = [
        'mnist',
        'fashion_mnist',
        'emnist_letters',
        'sign_lang',
        'clevr',
        'noise',
        'constant'
    ]
    num_filters = 32

# model hyperparameters as mentioned in Appendix A
mutation_rate = 0
reg_weight = 0
num_resnet = 2
num_hierarchies = 4
num_logistic_mix = 5
num_filters = num_filters
dropout_p = 0.3
learning_rate = 1e-3
use_weight_norm = True
epochs = 100
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
model_dir = '../saved_models/' + train_set + '/'

# load model
dist = network.PixelCNN(
      image_shape=input_shape,
      num_resnet=num_resnet,
      num_hierarchies=num_hierarchies,
      num_filters=num_filters,
      num_logistic_mix=num_logistic_mix,
      dropout_p=dropout_p,
      use_weight_norm=use_weight_norm,
      reg_weight=reg_weight
)

image_input = tfkl.Input(shape=input_shape)
log_prob = dist.log_prob(image_input)
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))
model.compile(optimizer=optimizer)

model.build([None] + list(input_shape))
model.load_weights(model_dir+'weights')

probs = {}
probs_3mad = {}
    
save_dir = '../probs_train_set/'

# get training set LL for conditional correction
if os.path.exists(save_dir + train_set + '.mat'):
    train_ll = loadmat(save_dir + train_set + '.mat')
    train_ll = train_ll['train_ll']
else:
    ds_train, _, _ = dataset_utils.get_dataset(
          train_set,
          512,
          mode,
          mutation_rate=mutation_rate
      )
    train_ll = []
    for batch in tqdm.tqdm(ds_train):
        train_ll.append(dist.log_prob(batch, training=False))
    train_ll = np.concatenate(train_ll, axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savemat(save_dir + train_set + '.mat', {'train_ll': train_ll})
    
threshold = np.median(train_ll)-3*stats.median_abs_deviation(np.squeeze(train_ll),scale='normal')
    
for dataset in datasets:
    
    print(dataset)
    
    _, _, ds_test = dataset_utils.get_dataset(
          dataset,
          512,
          mode,
          mutation_rate=mutation_rate
      )
    
    tmp = []
    for i in range (shuffles):
        globals()['tmp'+str(i)] = []
        
    for test_batch in tqdm.tqdm(ds_test):
        batch = tf.cast(test_batch, tf.float32).numpy()
        patches = patch_shuffle(batch, n_patches, orders) # get patch shuffled images
        tmp.append(dist.log_prob(batch, training=False).numpy())
        
        # get LL for shaken images
        for i in range (shuffles):
            globals()['tmp'+str(i)].append(dist.log_prob(np.round(patches[:,:,:,:,i]), training=False).numpy())
            
    tmp = np.expand_dims(np.concatenate(tmp, axis=0),axis=-1)
    for i in range (shuffles):
        globals()['tmp'+str(i)] = np.expand_dims(np.concatenate(globals()['tmp'+str(i)], axis=0),axis=-1)

    # add the differences of the log probabilities
    for i in range (shuffles):
        probs[dataset+'_shuffle_'+str(i)] = globals()['tmp'+str(i)]
        if i == 0:
            probs[dataset] = tmp - globals()['tmp'+str(i)]
        else:
            probs[dataset] += tmp - globals()['tmp'+str(i)]
            
    probs[dataset+'_regular'] = tmp
    
    # conditional correction using threshold
    probs_3mad[dataset] = probs[dataset].copy()
    probs_3mad[probs[dataset+'_regular']<threshold] = -1e10

save_dir = '../probs_shake/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
savemat(save_dir + train_set + '.mat', probs)

save_dir = '../probs_shake_3mad/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
savemat(save_dir + train_set + '.mat', probs_3mad)