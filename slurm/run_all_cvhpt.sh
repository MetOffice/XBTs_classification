
export XBT_INPUT_PATH=/scratch/shaddad/xbt-data/csv_with_imeta/

export BATCH_PREFIX=journal_paper_202104
export XBT_OUTPUT_PATH=/scratch/shaddad/xbt-data/${BATCH_PREFIX}/
export XBT_LOG_DIR=/data/users/shaddad/xbt_logs/${BATCH_PREFIX}

export XBT_REPO_ROOT=/home/h01/shaddad/prog/XBTs_classification/
export XBT_LAUNCH_SCRIPT=${XBT_REPO_ROOT}/slurm/launch_spice_experiment

export JOB_TIME=350
export XBT_QUEUE=normal
export EXEC_NAME=run_cvhpt_experiment
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_decisionTree_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_decisionTree_countryLatLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_decisionTree_latLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_decisionTree_latLon.json

export XBT_QUEUE=long
export JOB_TIME=720
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_randomForest_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_randomForest_countryLatLon.json

export JOB_TIME=1440
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_mlp_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_mlp_countryLatLon.json

export JOB_TIME=2160
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_knn_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_knn_countryLatLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} ${BATCH_PREFIX}_logreg_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/experiments/xbt_param_logreg_countryLatLon.json
