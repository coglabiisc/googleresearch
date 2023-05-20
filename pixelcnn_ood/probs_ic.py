import argparse
import tensorflow as tf
import numpy as np
import os
import tqdm
from scipy.io import savemat
from scipy.io import loadmat

from utils.dataset_utils import get_dataset
from utils.config_utils import import_config

tf.random.set_seed(42)

def main(args):

    train_set = args.train_set
    mode = args.mode
    compression = args.compression

    config = import_config()

    if compression == 'jpeg':
        from utils.ic_utils import find_jpeg_size as compress_function
    elif compression == 'png':
        from utils.ic_utils import find_png_size as compress_function
    elif compression == 'flif':
        from utils.ic_utils import find_flif_size as compress_function

    datasets = config['datasets'][mode]
        
    probs = {}

    # load log probabilities
    probs_ll = loadmat('../probs/' + train_set + '.mat')
        
    for dataset in datasets:

        _, _, ds_test = get_dataset(
            dataset,
            512,
            mode,
            mutation_rate=0
        )
        
        tmp = []
        
        # get compressed lengths for the test samples
        for test_batch in tqdm.tqdm(ds_test):
            tmp.append(compress_function(test_batch.numpy()))

        tmp = np.expand_dims(np.concatenate(tmp, axis=0),axis=-1)
        probs[dataset+'_size'] = tmp
        
        # compute IC score
        probs[dataset] = -(probs_ll[dataset]/np.log(2)) - probs[dataset+'_size']

    save_dir = '../probs_ic_' + compression + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    savemat(save_dir + train_set + '.mat', probs)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, default='mnist', help='train set name')
    parser.add_argument('--mode', type=str, default='grayscale', help='mode can be either grayscale or color')
    parser.add_argument('--compression', type=str, default='png', help='mode can be jpeg, png, or flif')
    args = parser.parse_args()
    
    main(args)