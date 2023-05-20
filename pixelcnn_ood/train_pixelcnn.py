import argparse
import tensorflow as tf
import numpy as np
import os
import tqdm
from scipy.io import savemat
from scipy.io import loadmat

from utils.dataset_utils import get_dataset
from utils.network import PixelCNN
from utils.checkpoint_utils import CNSModelCheckpoint
from utils.config_utils import import_config

tf.random.set_seed(42)

tfk = tf.keras
tfkl = tf.keras.layers

def main(args):

    train_set = args.train_set
    mode = args.mode
    bg = args.bg

    config = import_config()

    input_shape = tuple(config['input_shape'][mode])
    datasets = config['datasets'][mode]
        
    if bg:
        pre = 'bg/'
    else:
        pre = ''
        
    # model hyperparameters as mentioned in Appendix A

    reg_weight = config['reg_weight']
    use_weight_norm = config['use_weight_norm']
    learning_rate = config['learning_rate']
    num_resnet = config['num_resnet']
    num_hierarchies = config['num_hierarchies']
    num_logistic_mix = config['num_logistic_mix']
    num_filters = config['num_filters'][mode]
    dropout_p = config['dropout_p']
    epochs = config['epochs']
    batch_size = config['batch_size']
    mutation_rate = config['mutation_rate'][str(bg)][mode]
    test_batch_size = config['test_batch_size']

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
    model_dir = '../saved_models/' + pre + '/' + train_set + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # load train and validation sets
    ds_train, ds_val, _ = get_dataset(
        train_set,
        batch_size,
        mode,
        mutation_rate=mutation_rate
    )

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

    # fit and checkpoint the model
    model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_val,
            callbacks=[
                CNSModelCheckpoint(
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
        
        _, _, ds_test = get_dataset(
            dataset,
            test_batch_size,
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, default='mnist', help='train set name')
    parser.add_argument('--mode', type=str, default='grayscale', help='mode can be either grayscale or color')
    parser.add_argument('--bg', type=int, default=0, help='0 for regular model, 1 for background model for LRat')
    args = parser.parse_args()
    
    main(args)