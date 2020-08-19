import argparse
import os
import time
import xbt.common
from classification import experiment


def _get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    help_msg = 'The path to the JSON file containing the experiment definition.'
    parser.add_argument('--json-experiment', dest='json_experiment', help=help_msg)
    help_msg = ('The path to the directory containing the XBT dataset in csv '
                'form, one file per year. If --preproc-path is defined, these '
                'input files will be created by the preprocessing step and '
                'written to this location.')
    parser.add_argument('--input-path', dest='input_path', help=help_msg)
    help_msg = 'The path to the directory containing files to be preprocessed, typically netCDF files from WOD or similar source.'
    parser.add_argument('--preproc-path', dest='preproc_path', help=help_msg, default=None)
    help_msg = 'The path to the directory for experimenting output. A subdirectory will be created using the experiment name.'
    parser.add_argument('--output-path', dest='output_path', help=help_msg)
    help_msg = 'Specify whether classification output should be in a single file, or split by year or month.'
    parser.add_argument('--output-file-split', 
                        dest='output_file_split', 
                        help=help_msg, 
                        choices=xbt.common.OUTPUT_FREQS,
                        default = xbt.common.OUTPUT_SINGLE,
                       )
    return parser.parse_args()

def experiment_timer(exp_func):
    def run_experiment():
        start1 = time.time()
        ret_code = exp_func()
        end1 = time.time()
        duration1 = end1-start1
        print(f'experiment duration: {duration1:.3f} seconds')
        return ret_code
    return run_experiment
        
@experiment_timer
def run_single_experiment():
    """
    """
    exp_args = _get_arguments(
        'Run training, inference and evaluation on a single split.'
    )
    return_code = 0
    xbt_exp = experiment.ClassificationExperiment(exp_args.json_experiment, 
                                                         exp_args.input_path, 
                                                         exp_args.output_path,
                                                  preproc_dir=exp_args.preproc_path,
                                                  output_split=exp_args.output_file_split,
                                                 )
    try:
        xbt_exp.run_single_experiment()
    except RuntimeError as e1:
        print(f'Runtime error:\n {str(e1)}')
        return_code = 1
    return return_code


@experiment_timer
def run_cv_experiment():
    """
    """
    exp_args = _get_arguments(
        'Run training, inference and evaluation on multiple splits '
        'using cross-validation.'
    )
    return_code = 0
    xbt_exp = experiment.ClassificationExperiment(exp_args.json_experiment, 
                                                         exp_args.input_path, 
                                                         exp_args.output_path,
                                                  preproc_dir=exp_args.preproc_path,
                                                  output_split=exp_args.output_file_split,
                                                 )
    try:
        xbt_exp.run_cv_experiment()
    except RuntimeError as e1:
        print(f'Runtime error:\n {str(e1)}')
        return_code = 1
    
    return return_code

@experiment_timer
def run_cvhpt_experiment():
    """
    """
    exp_args = _get_arguments(
        'Run training, inference and evaluation on multiple splits,'
        'with hypterparameter tuning for each split and inner '
        'cross-validation on each set of parameters.'
    )
    return_code = 0
    xbt_exp = experiment.ClassificationExperiment(exp_args.json_experiment, 
                                                         exp_args.input_path, 
                                                         exp_args.output_path,
                                                  preproc_dir=exp_args.preproc_path,
                                                  output_split=exp_args.output_file_split,
                                                 )
    try:
        xbt_exp.run_cvhpt_experiment()
    except RuntimeError as e1:
        print(f'Runtime error:\n {str(e1)}')
        return_code = 1
    
    return return_code

@experiment_timer
def run_inference():
    """
    """
    exp_args = _get_arguments(
        'Run inference using previously trained classifiersand evaluation on a single split.'
    )
    return_code = 0
    xbt_exp = experiment.ClassificationExperiment(exp_args.json_experiment, 
                                                  exp_args.input_path, 
                                                  exp_args.output_path,
                                                  preproc_dir=exp_args.preproc_path,
                                                  output_split=exp_args.output_file_split,
                                                 )
    try:
        xbt_exp.run_inference()
    except RuntimeError as e1:
        print(f'Runtime error:\n {str(e1)}')
        return_code = 1
    
    return return_code
    