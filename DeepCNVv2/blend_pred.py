import numpy as np
import pandas as pd
import os
import sys
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, LeakyReLU, Lambda, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score

from numpy.random import seed
import matplotlib.pyplot as plt


JPG_dir = sys.argv[1]
metadata_dir = sys.argv[2]
metadata_file = sys.argv[3]
val_id_file = sys.argv[4]
model_name = sys.argv[5]
result_file = sys.argv[6]

metadata = metadata_dir + metadata_file

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(900,900), n_channels=3,
                  shuffle=True, meta_file='', meta_len = 13):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.meta_file = meta_file
        self.meta_len = meta_len
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        meta_df = pd.read_csv(self.meta_file)
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y= self.__data_generation(list_IDs_temp, meta_df)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp, meta_df):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X_image = np.empty((len(list_IDs_temp), self.dim[0], self.dim[1], self.n_channels))
        X_meta = np.empty((len(list_IDs_temp), self.meta_len), dtype=float)
        y = np.empty((len(list_IDs_temp)), dtype=int)
        meta_df = meta_df.drop(['length_ind'], axis = 1)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(JPG_dir + ID)
            img = cv2.resize(img,self.dim)
            if img is not None:
                X_image[i,] = img/255
                X_meta[i,] = meta_df[meta_df.id == ID].values[0][2:]
                y[i] = meta_df[meta_df.id == ID].values[0][1]

        return [X_image, X_meta], y

# Parameters
params_val = {'dim': (900, 900),
          'batch_size': 10,
          'n_channels': 3,
          'shuffle': False,
          'meta_file' : metadata,
          'meta_len' : 13}

# Datasets
valid_id = pd.read_csv(metadata_dir + val_id_file)

# Generators
valid_generator = DataGenerator(valid_id.id.values, valid_id.label.values, **params_val)

model1 = load_model(model_name)
y_prob = model1.predict(valid_generator)
y_prob = y_prob.flatten()
y_pred = [0 if x < 0.5 else 1 for x in y_prob]
auc_s = roc_auc_score(valid_id.label.values, y_prob)
print('auc: ', auc_s)

valid_id['y_prob'] = y_prob
valid_id['y_pred'] = y_pred
valid_id.to_csv(result_file, index = False)

