#!/bin/bash -l
#SBATCH --mem=20G
#SBATCH --ntasks=4
#SBATCH --output= YOUR_OUTPUT_FILE
#SBATCH --error= YOUR_ERROR_FILE
#SBATCH --time=01:00:00

# Caveats: values for memory size and timewall should be changed if needed

MYPATH=                                                # folder containig all training and test datasets
OUTPATH=                                               # folder where all the results should be stored
YEAR=                                                  # year to be processed
TRAIN=                                                 # train dataset name
TEST=                                                  # test dataset name
DESCRIPTOR=                                            # json file describing a specific experiment

python2.7 -m classification.classification --path $MYPATH --outpath $OUTPATH --year $YEAR $TRAIN $TEST $DESCRIPTOR
 
