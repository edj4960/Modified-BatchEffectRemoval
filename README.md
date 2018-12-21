# Calibration

repository for the paper "Removal of Batch Effects using Distribution-Matching Residual Networks" by Uri Shaham, Kelly P. Stanton, Jun Zhao, Huamin Li, Khadir Raddassi, Ruth Montgomery, and Yuval Kluger.

The script Train_MMD_ResNet.py is the main demo script, and can be generally used for calibration experiments. It was used to train all MMD ResNets used for the CyTOF experiments reported in our manuscript.
It loads two CyTOF datasets, corresponding to measurements of blood of the same person on the same machine in two different days, and denoises them. The script then trains a MMD-ResNet using one of the datasets as source and the other as target, to remove the batch effects. 

The CyTOF datasets used to produce the results in the manuscript are saved in Data.
The labels for the CyTOF datasets (person_Day_) were used only to separate the CD8 population during evaluation. Training of all models was unsupervised.
The RNA data set Data2_standardized_37PCs.csv contains the projection of the cleaned and filtered data onto the subspace of the first 37 principal components. To obtain the raw data please contact Jun Zhao at jun.zhao@yale.edu.  

All the models used to produce the results in the manuscript are saved in savedModels.

Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.

# Modifications

## Introduction
Modifications to this code have been made in order for easy testing between outputs of this code and [Confounded](https://github.com/jdayton3/Confounded) (a similar batch effect remover )
This code has been modified to take in one two datasets that have been stored in one csv file in the same format as Confounded (does

## How To Use 

To run this code, run the following command:

```bash
python3 train_MMD_ResNet.py
```

To alter the codes behavior, adjust the following variables in the command line:

| Variable         | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| `files`          | Name of source file and output output file.                               |
| `denoise`        | (Optional) option to train a denoising autoencoder to remove zeros. cmd: -d  |

Example:
```bash
python3 train_MMD_ResNet.py input.csv output -d
```

## Data
The given data to run is named 'tidy_batches_balanced.csv'. The code cannot run the full 'tidy_batches_balanced' so reduced tables are also added which contain the amount of variables in the title (ex. tidy_batches_balanced_2500.csv). These files are in the 'Data' folder.

There are also completed tests that are named output with the equivalent amount of variables in the title (ex. output_2500.csv). These files are in the BatchEffectRemoval-master folder.

Note that runtime can be especially long for large files. With 2500 variables expect around 3 hours to completion.
