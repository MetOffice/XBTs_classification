#!/bin/bash -l

module load scitools/experimental-current
echo "Running xbt experiment"
${XBT_REPO_ROOT}/bin/run_cvhpt_experiment --json-experiment ${XBT_JSON_EXP} --input-path ${XBT_INPUT_PATH} --output-path ${XBT_OUTPUT_PATH}