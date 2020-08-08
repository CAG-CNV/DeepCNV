# DeepCNV: A Deep Learning Approach for Authenticating Copy Number Variants
For any question about this repo, please contact Xiurui Hou (xh256@njit.edu).  

## Description
We propose a deep learning approach to remove the false positive CNV calls from SNP array and sequencing CNV detection programs. This repo constains the model code and an executable script with five sample inputs. Since the pre-trained model file exceeds the upload size of Github, it can be accessed by this external [link](https://www.filehosting.org/file/details/886348/DeepCNV.hdf5). The dataset of this project is not for public. blended_learning.py is the training script. You can feed your own dataset to train the model using blended_learning.py.

## Generate plot images for script
```perl visualize_cnv.pl -format plot -signal 200477520001_R06C01.baflrr 200477520001_R06C01.rawcnv```;  
Typically the baflrr signal file has header: Name Chr Position sample.B Allele Freq sample.Log R Ratio;  
```--snpposfile NameChrPosition.txt``` can be added if only "Name sample.B Allele Freq sample.Log R Ratio" columns provided in baflrr signal files;  
PennCNV-Seq can be run on sequencing BAM/CRAM to generate baflrr files;  
Rawcnv input file is: chr:start-stop numsnp=1 length=1 state2,cn=1 200477520001_R06C01.baflrr startsnp=a endsnp=b;  
chr:start-stop and 200477520001_R06C01.baflrr are the only critical fields to be specified, making it easily adaptable to most CNV call output formats;  

## Run script
1. Download the pre-trained model file from this [link](https://www.filehosting.org/file/details/886348/DeepCNV.hdf5);
2. Download script folder;
3. Copy model file into script folder;
4. Enter script folder from Terminal;
5. Check the package requirments. Different package may generate different results;
6. Create output folder by ```mkdir output```;
7. Run ```python run.py ./data/JPG ./data/samples.csv ./output ./DeepCNV.hdf5```;
8. Check the results from output folder.

## Package Requirments
python  2.7.12  
pandas  0.17.1  
numpy  1.11.0  
tensorflow  1.12.0  
keras  2.2.4  
cv2  2.4.9.1  

