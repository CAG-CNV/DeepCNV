import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support
import matplotlib.pyplot as plt
import sys, os

JPG_dir = sys.argv[1]
model_name = sys.argv[2]
result_file = sys.argv[3]

batch_size = 64
target_size = (300, 300)
params = dict(rescale=1./255,
                horizontal_flip=True,
                fill_mode='nearest')

train_datagen = ImageDataGenerator(**params)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        os.path.join(JPG_dir,'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
        os.path.join(JPG_dir,'val'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        os.path.join(JPG_dir,'test'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

def model1():
    inputs = Input(shape=target_size+(3,))
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(0.1)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPool2D((2,2))(conv1)
    
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = LeakyReLU(0.1)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPool2D((2,2))(conv2)
    
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = LeakyReLU(0.1)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPool2D((2,2))(conv3)
    
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = LeakyReLU(0.1)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPool2D((2,2))(conv4)
    
    dense1 = GlobalAveragePooling2D()(conv4)
    
    dense1 = Dense(50, activation= 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                  kernel_initializer='he_normal')(dense1)
    outputs = Dense(1, activation= 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                  kernel_initializer='he_normal')(dense1)
    
    model = tf.keras.Model(inputs, outputs)
    return model

tf.keras.backend.clear_session()
model = model1()
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0),
              metrics=['accuracy', 'AUC'])

earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=0)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),min_lr=0.5e-6,
                                           cooldown=0, patience=2, verbose = 1)

history = model.fit(x = train_generator,
                    epochs=1,
                    verbose=1,
                    validation_data = validation_generator, 
                    callbacks=[earlystopper,lr_reducer])

model.save(model_name)

y_prob = model.predict(test_generator)
y_prob = y_prob.flatten()
y_pred = (y_prob>=0.5).astype(int)
auc_s = roc_auc_score(test_generator.classes, y_prob)
print('auc: ', auc_s)

df = pd.DataFrame()
df['files'] = test_generator.directory
df['y_prob'] = y_prob
df.to_csv(result_file)