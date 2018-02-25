#!/bin/bash -l
#SBATCH --mem=20G
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

# Caveats: values for memory size and timewall should be changed if needed

MYPATH=                                                # folder containig all training and test datasets
OUTPATH=                                               # folder where all the results should be stored
YEAR=                                                  # year to be processed: when many experiments for several years are processed together, then this variable is not used, since it is inherited from the parent script `multi_submission_pipeline.sh`
TRAIN=                                                 # train dataset name
TEST=                                                  # test dataset name
DESCRIPTOR=                                            # json file describing a specific experiment: when many experiments for several years are processed together, then this variable defines e template, e.g. `experiment_${NUMBER}.json`, where ${NUMBER} is inherited from the parent script `multi_submission_pipeline.sh`

python2.7 -m classification.classification --path $MYPATH --outpath $OUTPATH --year $YEAR $TRAIN $TEST $DESCRIPTOR
 
