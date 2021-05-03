1. foder structure
    ./data/                   # data samples
    ./data/train/1/           # Positive training images samples
    ./data/train/0/           # Negative training images samples
    ./data/val/1/             # Positive validation images samples
    ./data/val/0/             # Negative validation images samples
    ./data/test/1/            # Positive testing images samples
    ./data/test/0/            # Negative testing images samples
    
    ./res                     # folder to store results

    ./best_model_seq.hdf5       # a pretrained model

2. To use the script to predict the images, follow the steps below:
    a. To train the model run "python train.py img_folder saved_model_name results_file"
    e.g.
    python blend.py data/ res/model.hdf5 res/res.csv

    b. To make prediction with a pretrained model run "python predict.py img_folder saved_model_name results_file"
    e.g.
    python predict.py data/ res/model.hdf5 res/res.csv

3. The output includes:
    a. trained model
    b. prediction for testing samples

4. The packages version:
    python      3.7.6
    tensorflow  2.1.0
