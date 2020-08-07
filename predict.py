import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support

import os
import sys

image_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(image_dir):
    print ("Input directory does not exist!")
    sys.exit(1)
if not os.path.exists(output_dir):
    print ("Output directory does not exist!")
    sys.exit(1)
output_pos_dir = output_dir + "/1"
output_neg_dir = output_dir + "/0"
if os.path.exists(output_pos_dir):
    os.system("rm " + output_pos_dir + "/*")
else:
    os.system("mkdir " + output_pos_dir)
if os.path.exists(output_neg_dir):
    os.system("rm " + output_neg_dir + "/*")
else:
    os.system("mkdir " + output_neg_dir)

model_name = 'model/best_model_1_1.hdf5'

batch_size = 64
target_size = (300, 300)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

print ("Loading model....")
model = tf.keras.models.load_model(model_name)
#model.summary()
print ("Predicting....")
prob = model.predict(test_generator, verbose=0)
pred = [0 if x < 0.5 else 1 for x in prob]
y_te = test_generator.classes
auc_test = roc_auc_score(y_te, prob)
mcc_test = matthews_corrcoef(y_te, pred)
prfs_test = precision_recall_fscore_support(y_te, pred)
acc_test = accuracy_score(y_te, pred)
print ('auc:', auc_test)
print ('mcc:', mcc_test)
print ('acc:', acc_test)
print ('negative ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs_test[0][0], prfs_test[1][0], prfs_test[2][0], prfs_test[3][0]))
print ('positive ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs_test[0][1], prfs_test[1][1], prfs_test[2][1], prfs_test[3][1]))

files = test_generator.filenames
res = pd.DataFrame(
    {'Id': files,
     'prediction': pred,
     'probability': prob.flatten()
    })
res.to_csv(output_dir + "/res.csv", index = False)

for i, f in enumerate(files):
    if pred[i] == 0:
        os.system("cp " + image_dir + "/" + f + " " + output_neg_dir)
    else:
        os.system("cp " + image_dir + "/" + f + " " + output_pos_dir)
print ("Finish...")