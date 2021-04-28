#!/bin/bash -l

module load scitools/experimental-current
echo "Running xbt experiment"
echo ${XBT_REPO_ROOT}/bin/${XBT_EXEC} --json-experiment ${XBT_JSON_EXP} --input-path ${XBT_INPUT_PATH} --output-path ${XBT_OUTPUT_PATH} ${XBT_PREPROC_ARG}
${XBT_REPO_ROOT}/bin/${XBT_EXEC} --json-experiment ${XBT_JSON_EXP} --input-path ${XBT_INPUT_PATH} --output-path ${XBT_OUTPUT_PATH} ${XBT_PREPROC_ARG}