import os
import sys
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
K.clear_session()

image_dir = sys.argv[1]
metadata_dir = sys.argv[2]
output_dir = sys.argv[3]
model_path = sys.argv[4]

if not os.path.exists(image_dir):
    print ("Input directory does not exist!")
    sys.exit(1)
if not os.path.exists(output_dir):
    print ("Output directory does not exist!")
    sys.exit(1)
output_pos_dir = output_dir + "/pos"
output_neg_dir = output_dir + "/neg"
if os.path.exists(output_pos_dir):
	os.system("rm " + output_pos_dir + "/*")
else:
	os.system("mkdir " + output_pos_dir)
if os.path.exists(output_neg_dir):
	os.system("rm " + output_neg_dir + "/*")
else:
	os.system("mkdir " + output_neg_dir)

input_files = [each for each in os.listdir(image_dir) if each.endswith('.JPG')]
print ("There are " + str(len(input_files)) + " input images.")

print ("Loading model....")
meta_df = pd.read_csv(metadata_dir)
model = load_model(model_path)

print ("Predicting....")
fnames = meta_df.ID.values
prob = []
pred = []
for f in fnames:
    x_image = cv2.imread(image_dir + '/' + f)/255
    x_image = np.expand_dims(x_image, axis=0)
    x_meta = meta_df[meta_df.ID == f].iloc[:, 1:14].values[0]
    x_meta = np.expand_dims(x_meta, axis=0)
    x = [x_image, x_meta]
    probability = model.predict(x)[0][0]
    prediction = 1 if probability >= 0.5 else 0
    prob.append(probability)
    pred.append(prediction)
    if prediction == 0:
        os.system("cp " + image_dir + "/" + f + " " + output_neg_dir)
    else:
        os.system("cp " + image_dir + "/" + f + " " + output_pos_dir)
print ("Finish...")

res = pd.DataFrame(
    {'Id': fnames,
     'prediction': pred,
     'probability': prob
    })

res.to_csv(output_dir + "/res.csv", index = False)

neg_pred = len([x for x in pred if x == 0])
pos_pred = len([x for x in pred if x == 1])
print ("Predicted negative: " + str(neg_pred))
print ("Predicted positive: " + str(pos_pred))