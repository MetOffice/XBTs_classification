#!/bin/bash

START=  # Starting year
END=    # Ending year

START_EXP= # First experiment number 
END_EXP= # Last experiment number

SEQUENCE=`seq ${START} ${END}`
N_EXPERIMENTS=`seq ${START_EXP} ${END_EXP}`

SINGLE_EXP_SCRIPT='experiment_submission_script.sh'

for year in ${SEQUENCE}
do

    export YEAR=$year

    for experiment in ${N_EXPERIMENTS}
    do
	export EXP_INDEX=$experiment
	echo  -e "Submission of experiment ${EXP_INDEX} for year ${YEAR}"
	sbatch < ${SINGLE_EXP_SCRIPT}
    done
done