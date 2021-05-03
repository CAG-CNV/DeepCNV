1. foder structure
    ./data/                   # data samples
    ./data/JPG                # CNV images samples
    ./data/metadata.csv       # metadata samples
    ./data/train_sample.csv   # training data id
    ../data/val_sample.csv    # validation data id
    
    ./res                     # folder to store results

    ./best_model_0.hdf5       # a pretrained model

2. To use the script to predict the images, follow the steps below:
    a. To train the model run "python blend.py img_folder metadata_folder metadata_file train_id_file val_id_file saved_model_name results_file"
    e.g.
    python blend.py data/JPG/ data/ metadata.csv train_sample.csv val_sample.csv res/bmodel.hdf5 res/res.csv

    b. To make prediction with a pretrained model run "python blend.py img_folder metadata_folder metadata_file val_id_file saved_model_name results_file"
    e.g.
    python blend_pred.py data/JPG/ data/ metadata.csv val_sample.csv best_model_0.hdf5 res/res.csv

3. The output includes:
    a. trained model
    b. prediction for validation samples

4. The packages version:
    python      3.7.6
    tensorflow  2.1.0
