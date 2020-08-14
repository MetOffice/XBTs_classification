export XBT_INPUT_PATH=/scratch/shaddad/xbt-data/csv_with_imeta/
export XBT_OUTPUT_PATH=/scratch/shaddad/xbt-data/cv_outputs_202008/
export XBT_LOG_DIR=/data/users/shaddad/xbt_logs/report_202008

export XBT_REPO_ROOT=/home/h01/shaddad/prog/XBTs_classification/
export XBT_LAUNCH_SCRIPT=${XBT_REPO_ROOT}/slurm/launch_spice_experiment

export JOB_TIME=350
export XBT_QUEUE=normal
export EXEC_NAME=run_cv_experiment
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_decisionTree_country ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_decisionTree_country.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_decisionTree_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_decisionTree_countryLatLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_decisionTree_latLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_decisionTree_latLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_decisionTree_maxDepthYear ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_decisionTree_maxDepthYear.json

export XBT_QUEUE=long
export JOB_TIME=720
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_randomForest_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_randomForest_countryLatLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_randomForest_country ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_randomForest_country.json

export JOB_TIME=1440
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_mlp_country ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_mlp_country.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_mlp_countryLatLon ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_mlp_countryLatLon.json
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_knn_country ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_knn_country.json

export JOB_TIME=2160
${XBT_LAUNCH_SCRIPT} ${EXEC_NAME} report_sa8_logreg_country ${XBT_REPO_ROOT}  ${XBT_REPO_ROOT}/examples/xbt_param_logreg_country.json
