import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import os
import sys
import tqdm
from scipy.io import savemat
from scipy.io import loadmat

sys.path.append('../Dependencies/')
import dataset_utils
import network
import utils

tf.random.set_seed(42)

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

train_set = 'mnist'
mode = 'grayscale'
bg = 0 # 0 - regular model, 1 - background model for LRat

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
    if bg:
        mutation_rate = 0.1
    else:
        mutation_rate = 0
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
    if bg:
        mutation_rate = 0.3
    else:
        mutation_rate = 0
    
if bg:
    pre = 'bg/'
else:
    pre = ''
    
# model hyperparameters as mentioned in Appendix A
reg_weight = 0
use_weight_norm = True
learning_rate = 1e-3
num_resnet = 2
num_hierarchies = 4
num_logistic_mix = 5
num_filters = num_filters
dropout_p = 0.3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
epochs = 100
batch_size = 32
    
model_dir = '../saved_models/' + pre + '/' + train_set + '/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
# load train and validation sets
ds_train, ds_val, _ = dataset_utils.get_dataset(
      train_set,
      batch_size,
      mode,
      mutation_rate=mutation_rate
  )

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

# fit and checkpoint the model
model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[
            utils.CNSModelCheckpoint(
                filepath=os.path.join(model_dir+'weights'),
                verbose=1, save_weights_only=True, save_best_only=True
                )
        ],
    )

# load best weights
model.load_weights(model_dir+'weights')

probs = {}

# load foreground probabilities for LRat background model
if bg:
    probs_fg = loadmat('../probs/' + train_set + '.mat')
        
for dataset in datasets:
    
    _, _, ds_test = dataset_utils.get_dataset(
          dataset,
          256,
          mode,
          mutation_rate=0
      )
    tmp = []
    
    # compute LL for test samples
    for test_batch in tqdm.tqdm(ds_test):
        tmp.append(dist.log_prob(tf.cast(test_batch, tf.float32),
                                        training=False).numpy())

    tmp = np.expand_dims(np.concatenate(tmp, axis=0),axis=-1)
    tmp = np.array(tmp)

    if bg:
        # Likelihood ratio
        probs[dataset] = probs_fg[dataset] - tmp
    else:
        probs[dataset] = tmp

save_dir = '../probs/' + pre + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

savemat(save_dir + train_set + '.mat', probs)