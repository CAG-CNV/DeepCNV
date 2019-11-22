# DeepCNV: A Deep Learning Approach for Authenticating Copy Number Variants
For any question about this repo, please contact Xiurui Hou (xh256@njit.edu).  

## Description
We propose a deep learning approach to remove the false positive CNV calls from PennCNV program. This repo constains the model code and an executable script with five sample inputs. Since the model file exceeds the upload size of Github, it can be accessed by this external [link](https://www.filehosting.org/file/details/833894/DeepCNV.hdf5). The dataset of this project is not for public. blended_learning.py is the training script.

## Run script
1. Download the model file from this [link](https://www.filehosting.org/file/details/833894/DeepCNV.hdf5);
2. Download script folder;
3. Copy model file into script folder;
4. Enter script folder from Terminal;
5. Check the package requirments. Different package may generate different results;
6. Create output folder by ```mkdir output```;
7. Run ```python run.py ./data/JPG ./data/samples.csv ./output ./DeepCNV.hdf5```;
8. Check the results from output folder.

## Package Requirments
Python  2.7.12  
pandas  0.17.1  
numpy  1.11.0  
tensorflow  1.12.0  
keras  2.2.4  
cv2  2.4.9.1  

