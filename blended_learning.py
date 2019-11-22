import numpy as np
import pandas as pd
import os
import cv2

import keras
from keras import backend as K
from keras.models import Sequential
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support

from numpy.random import seed
seed(1)
K.clear_session()
K.set_image_data_format('channels_last')

JPG_dir = sys.argv[1]
metadata_dir = sys.argv[2]
metadata = metadata_dir + 'metadata.csv'
model_name = sys.argv[3]
result_name = sys.argv[4]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(900,900), n_channels=3,
                 n_classes=2, shuffle=True, meta_file='', meta_len = 13):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.meta_file = meta_file
        self.meta_len = meta_len
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
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
        X_image = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        X_meta = np.empty((self.batch_size, self.meta_len), dtype=float)
        y = np.empty((self.batch_size), dtype=int)
        meta_df = meta_df.drop(['length_ind'], axis = 1)

        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(JPG_dir + ID)
            if img is not None:
              X_image[i,] = img/255
              X_meta[i,] = meta_df[meta_df.ID == ID].values[0][2:]
              y[i] = meta_df[meta_df.ID == ID].values[0][1]

        return [X_image, X_meta], y

# Parameters
params = {'dim': (900,900),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True,
          'meta_file' : metadata,
          'meta_len' : 13}

# Datasets
train_id = pd.read_csv(metadata_dir + 'train_id.csv')
valid_id = pd.read_csv(metadata_dir + 'valid_id.csv')
    
# Generators
train_generator = DataGenerator(train_id.id.values, train_id.label.values, **params)
valid_generator = DataGenerator(valid_id.id.values, valid_id.label.values, **params)

# first input model
visible1 = Input(shape=(900,900,3))
conv11 = Conv2D(32, kernel_size=3, activation='linear')(visible1)
leaky1 = LeakyReLU(alpha=.001)(conv11)
pool11 = MaxPooling2D(pool_size=(2, 2))(leaky1)

conv12 = Conv2D(64, kernel_size=3, activation='linear')(pool11)
leaky2 = LeakyReLU(alpha=.001)(conv12)
pool12 = MaxPooling2D(pool_size=(2, 2))(leaky2)

conv13 = Conv2D(64, kernel_size=3, activation='linear')(pool12)
leaky3 = LeakyReLU(alpha=.001)(conv13)
pool13 = MaxPooling2D(pool_size=(2, 2))(leaky3)

conv14 = Conv2D(128, kernel_size=3, activation='linear')(pool13)
leaky4 = LeakyReLU(alpha=.001)(conv14)
pool14 = MaxPooling2D(pool_size=(2, 2))(leaky4)

conv15 = Conv2D(128, kernel_size=3, activation='linear')(pool14)
leaky5 = LeakyReLU(alpha=.001)(conv15)
pool15 = MaxPooling2D(pool_size=(2, 2))(leaky5)

conv16 = Conv2D(256, kernel_size=3, activation='linear')(pool15)
leaky6 = LeakyReLU(alpha=.001)(conv16)
pool16 = MaxPooling2D(pool_size=(2, 2))(leaky6)

dropout1 = Dropout(0.4)(pool16)
flat1 = Flatten()(dropout1)

# second input model
visible2 = Input(shape=(13,))
dense1 = Dense(36, activation='relu')(visible2)
dense2 = Dense(24, activation='relu')(dense1)
dense3 = Dense(24, activation='relu')(dense2)
dense4 = Dense(12, activation='relu')(dense3)
dropout2 = Dropout(0.4)(dense4)

# merge input models
merge = concatenate([flat1, dropout2])
# interpretation model
hidden1 = Dense(50, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=[visible1, visible2], outputs=output)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0),
              metrics=['acc'])

# early stopping
checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

# Train model on dataset
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=30,
                    epochs=100,
                    verbose = 1,
                    validation_data=valid_generator,
                    validation_steps=30,
                    callbacks=[checkpointer,earlystopper])

model = load_model(model_name)

# Test model on test dataset
test_id = pd.read_csv(metadata_dir + 'test_id.csv')
meta_df = pd.read_csv(metadata)
truth = test_id.label.values
fnames = test_id.id.values
prob = []
length_ind = []

print ('start predict....')
for index, row in test_id.iterrows():
    f = row[0]
    x_image = cv2.imread(JPG_dir + f)/255
    x_image = np.expand_dims(x_image, axis=0)
    x_meta = meta_df[meta_df.ID == f].iloc[:, 2:15].values[0]
    length_ind.append(meta_df[meta_df.ID == f].iloc[:, 15].values[0])
    x_meta = np.expand_dims(x_meta, axis=0)
    x = [x_image, x_meta]
    prob.append(model.predict(x, verbose=0)[0][0])

print ('end predict...')
pred = [0 if x < 0.5 else 1 for x in prob]
auc_test = roc_auc_score(truth, prob)
mcc_test = matthews_corrcoef(truth, pred)
prfs_test = precision_recall_fscore_support(truth, pred)
dl_result = pd.DataFrame({'fname':fnames, 'truth':truth, 'dl_prob':prob, 'dl_pred':pred, 'length_ind':length_ind})
correct = dl_result['dl_pred'] == dl_result['truth']
acc_test = float(sum(correct)) / len(dl_result['truth'])
print ('auc:', auc_test)
print ('mcc:', mcc_test)
print ('acc:', acc_test)
print ('negative ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs_test[0][0], prfs_test[1][0], prfs_test[2][0], prfs_test[3][0]))
print ('positive ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs_test[0][1], prfs_test[1][1], prfs_test[2][1], prfs_test[3][1]))
dl_result.to_csv(result_name, index = False)