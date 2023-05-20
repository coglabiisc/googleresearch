import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tqdm
from scipy.io import savemat
from scipy.io import loadmat
import os
from scipy import stats

from utils.dataset_utils import get_dataset
from utils.network import PixelCNN
from utils.transform_utils import transform
from utils.config_utils import import_config

tf.random.set_seed(42)

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

def main(args):

    train_set = args.train_set
    mode = args.mode

    config = import_config()

    input_shape = tuple(config['input_shape'][mode])
    datasets = config['datasets'][mode]

    # model hyperparameters as mentioned in Appendix A
    mutation_rate = 0
    reg_weight = config['reg_weight']
    use_weight_norm = config['use_weight_norm']
    learning_rate = config['learning_rate']
    num_resnet = config['num_resnet']
    num_hierarchies = config['num_hierarchies']
    num_logistic_mix = config['num_logistic_mix']
    num_filters = config['num_filters'][mode]
    dropout_p = config['dropout_p']
    test_batch_size = config['test_batch_size']

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
    model_dir = '../saved_models/' + train_set + '/'

    # load model
    dist = PixelCNN(
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

    rots = ['rot90', 'rot180', 'rot270', 'inv', 'rot90_inv', 'rot180_inv', 'rot270_inv']

    probs = {}
    probs_3mad = {}

    save_dir = '../probs_train_set/'

    # get training set LL for conditional correction
    if os.path.exists(save_dir + train_set + '.mat'):
        train_ll = loadmat(save_dir + train_set + '.mat')
        train_ll = train_ll['train_ll']
    else:
        ds_train, _, _ = get_dataset(
            train_set,
            test_batch_size,
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
        
        _, _, ds_test = get_dataset(
            dataset,
            test_batch_size,
            mode,
            mutation_rate=mutation_rate
        )
        
        tmp = []
        for rot in rots:
            globals()['tmp_'+rot] = []
        for test_batch in tqdm.tqdm(ds_test):
            batch = tf.cast(test_batch, tf.float32).numpy()
            tmp.append(dist.log_prob(batch, training=False).numpy())
            
            # get LL for stirred images
            for j, rot in enumerate(rots):
                temp = transform(batch, j)
                globals()['tmp_'+rot].append(dist.log_prob(temp, training=False).numpy())

        tmp = np.expand_dims(np.concatenate(tmp, axis=0),axis=-1)
        for rot in rots:
            globals()['tmp_'+rot] = np.expand_dims(np.concatenate(globals()['tmp_'+rot], axis=0),axis=-1)

        probs[dataset+'_regular'] = tmp

        # add the differences of the log probabilities
        for j, rot in enumerate(rots):
            probs[dataset+'_'+rot] = globals()['tmp_'+rot]
            if j == 0:
                probs[dataset] = (tmp - globals()['tmp_'+rot])
            else:
                probs[dataset] += (tmp - globals()['tmp_'+rot])
                
        # conditional correction using threshold
        probs_3mad[dataset] = probs[dataset].copy()
        probs_3mad[probs[dataset+'_regular']<threshold] = -1e10

    save_dir = '../probs_stir/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    savemat(save_dir + train_set + '.mat', probs)

    save_dir = '../probs_stir_3mad/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    savemat(save_dir + train_set + '.mat', probs_3mad)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, default='mnist', help='train set name')
    parser.add_argument('--mode', type=str, default='grayscale', help='mode can be either grayscale or color')
    args = parser.parse_args()
    
    main(args)