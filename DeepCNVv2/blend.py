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
train_id_file = sys.argv[4]
val_id_file = sys.argv[5]
model_name = sys.argv[6]
result_file = sys.argv[7]

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
params_tr = {'dim': (900, 900),
          'batch_size': 10,
          'n_channels': 3,
          'shuffle': True,
          'meta_file' : metadata,
          'meta_len' : 13}

params_val = {'dim': (900, 900),
          'batch_size': 10,
          'n_channels': 3,
          'shuffle': False,
          'meta_file' : metadata,
          'meta_len' : 13}

# Datasets
train_id = pd.read_csv(metadata_dir + train_id_file)
valid_id = pd.read_csv(metadata_dir + val_id_file)

# Generators
train_generator = DataGenerator(train_id.id.values, train_id.label.values, **params_tr)
valid_generator = DataGenerator(valid_id.id.values, valid_id.label.values, **params_val)


K.clear_session()
# first input model
visible1  = Input(shape=params_tr['dim']+(3,))
conv1 = Conv2D(32, 3,  kernel_initializer='he_normal')(visible1)
conv1 = LeakyReLU(0.001)(conv1)
conv1 = MaxPool2D((2,2))(conv1)
    
conv2 = Conv2D(64, 3,  kernel_initializer='he_normal')(conv1)
conv2 = LeakyReLU(0.001)(conv2)
conv2 = MaxPool2D((2,2))(conv2)
    
conv2 = Conv2D(64, 3,  kernel_initializer='he_normal')(conv2)
conv2 = LeakyReLU(0.001)(conv2)
conv2 = MaxPool2D((2,2))(conv2)
    
conv3 = Conv2D(128, 3,  kernel_initializer='he_normal')(conv2)
conv3 = LeakyReLU(0.001)(conv3)
conv3 = MaxPool2D((2,2))(conv3)

conv3 = Conv2D(128, 3,  kernel_initializer='he_normal')(conv3)
conv3 = LeakyReLU(0.001)(conv3)
conv3 = MaxPool2D((2,2))(conv3)
    
conv4 = Conv2D(256, 3, kernel_initializer='he_normal')(conv3)
conv4 = LeakyReLU(0.001)(conv4)
conv4 = MaxPool2D((2,2))(conv4)

flat1 = Dropout(0.4)(conv4)
flat1 = Flatten()(flat1)

# second input model
visible2 = Input(shape=(13,))
dense1 = Dense(36, activation='relu')(visible2)
dense2 = Dense(24, activation='relu')(dense1)
dense3 = Dense(24, activation='relu')(dense2)
dense4 = Dense(12, activation='relu')(dense3)
dropout2 = Dropout(0.4)(dense4)

# merge input models
merge = Lambda(lambda x: tf.concat([x[0],x[1]], 1))([flat1, dropout2])
# interpretation model
hidden1 = Dense(50, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=[visible1, visible2], outputs=output)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0),
              metrics=['accuracy', 'AUC'])

# early stopping
checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=True)
#earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

# Train model on dataset
history = model.fit_generator(generator = train_generator,
                    steps_per_epoch = train_generator.__len__(),
                    epochs=6,
                    verbose = 1,
                    validation_data = valid_generator,
                    validation_steps = valid_generator.__len__(),
                    callbacks=[checkpointer])

print(np.argmin(history.history['val_loss']) )

model1 = load_model(model_name)
y_prob = model1.predict(valid_generator)
y_prob = y_prob.flatten()
y_pred = [0 if x < 0.5 else 1 for x in y_prob]
auc_s = roc_auc_score(valid_id.label.values, y_prob)
print('auc: ', auc_s)

valid_id['y_prob'] = y_prob
valid_id['y_pred'] = y_pred
valid_id.to_csv(result_file, index = False)

