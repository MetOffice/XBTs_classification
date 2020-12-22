#!/bin/bash -l
#SBATCH --mem=24G
#SBATCH --ntasks=4
#SBATCH --time=60
#SBATCH --output=xbt_wod_extract.out
#SBATCH --error=xbt_wod_extract.err

module load scitools/experimental-current
echo running command ${XBT_REPO_ROOT}/bin/wod_extract_years --input-path ${XBT_INPUT_PATH} --output-path ${XBT_OUTPUT_PATH} --num-tasks 4 --prefix ${XBT_PREFIX} --suffix ${XBT_SUFFIX}

${XBT_REPO_ROOT}/bin/wod_extract_years --input-path ${XBT_INPUT_PATH} --output-path ${XBT_OUTPUT_PATH} --num-tasks 4 --prefix ${XBT_PREFIX} --suffix ${XBT_SUFFIX}
