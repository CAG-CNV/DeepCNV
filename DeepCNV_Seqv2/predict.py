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
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        os.path.join(JPG_dir,'test'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

model = tf.keras.models.load_model(model_name)

y_prob = model.predict(test_generator)
y_prob = y_prob.flatten()
y_pred = (y_prob>=0.5).astype(int)
auc_s = roc_auc_score(test_generator.classes, y_prob)
print('auc: ', auc_s)

df = pd.DataFrame()
df['files'] = test_generator.directory
df['y_prob'] = y_prob
df.to_csv(result_file)