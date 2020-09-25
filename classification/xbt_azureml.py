import pathlib
import argparse
import os
import tempfile

import azureml.core 
import azureml.core.run
import dataexploration.xbt_dataset
import classification.experiment
import xbt.common

class AzureDataset(dataexploration.xbt_dataset.XbtDataset):
    def __init__(self, year_range, azml_ws, azml_dataset_name, 
                 df=None, use_dask=False, load_profiles=False, load_quality_flags=False, do_preproc_extract=False, pp_prefix='', pp_suffix='', pp_csv_dir=None):
        
        # Set up the azure dataset and mount the files in a temporary directory. This location
        # is then passed to the parent class which can proceed with loading data in the normal
        # way.
        self._azml_workspace = azml_ws
        self._azml_dataset_name = azml_dataset_name
        self.azml_dataset = azureml.core.Dataset.get_by_name(self._azml_workspace, 
                                                name=self._azml_dataset_name)
        self._input_mount = self.azml_dataset.mount()
        self._input_mount.start()    
        self.azml_mount_dir = str(self._input_mount.mount_point)
        super().__init__(self.azml_mount_dir, 
                                year_range, 
                                df, 
                                use_dask, 
                                load_profiles, 
                                load_quality_flags, 
                                do_preproc_extract,
                                pp_prefix,
                                pp_suffix,
                             pp_csv_dir
                      )
            
def to_azure_table(input_dict):
    return [dict(zip(['index'] + list(input_dict), record1)) for record1 in input_dict.to_records()]
    
class AzureExperiment(classification.experiment.ClassificationExperiment):
    
    def __init__(self, json_descriptor,data_dir, output_dir, output_split, do_preproc_extract=False):
        self._azml_run = azureml.core.run.Run.get_context()
        self._azml_experiment = self._azml_run.experiment
        self._azml_workspace = self._azml_experiment.workspace
#         self._azml_ds_name = input_dataset_name
        
# remove this once the externally defined (i.e. in the notebook) mounts are working
        self._temp_output = tempfile.TemporaryDirectory()
        
        super().__init__(json_descriptor=json_descriptor, 
                         data_dir=data_dir, 
                         output_dir=output_dir, 
                         output_split=output_split, 
                         do_preproc_extract=do_preproc_extract)
    
    def _write_exp_outputs_to_azml(self):
        # upload small files to the run
        if self.scores_out_path:
            self._azml_run.log_table('scores',
                                     {c1: list(self.score_table[c1]) for c1 in self.score_table.columns})
            print(f'uploading scores file {self.scores_out_path} to AzML run')
            self._azml_run.upload_file(os.path.split(self.scores_out_path)[1],
                                       self.scores_out_path)
        
        if self.metrics_out_path:
            print(f'uploading metrics file {self.metrics_out_path} to AzML run')
            self._azml_run.upload_file(os.path.split(self.metrics_out_path)[1],
                                       self.metrics_out_path)
            # only upload metrics for all classes, too much data otherwise for AzML
            metric_list = [c1 for c1 in self.results.columns if '_all' in c1]
#             for metric_name in metric_list:
#                 self._azml_run.log_table(f'metric_{metric_name}', {
#                     'year': list(self.results['year']),
#                     metric_name: list(self.results[metric_name])})
            
        for model_path1 in self.classifiers_export_path_list:
            model_upload_name = model_name=os.path.split(model_path1)[1]
            self._azml_run.upload_file(model_upload_name,
                                       model_path1,
                                      )
            self._azml_run.register_model(
                model_upload_name,
            )
        
    def run_single_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        super().run_single_experiment(write_results, write_predictions, export_classifiers)
        self._write_exp_outputs_to_azml()
        
    def run_cv_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        super().run_cv_experiment(write_results, write_predictions, export_classifiers)
        self._write_exp_outputs_to_azml()
        
    def run_cvhpt_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        super().run_cvhpt_experiment(write_results, write_predictions, export_classifiers)
        self._write_exp_outputs_to_azml()

    def run_inference(self, write_predictions=True):    
        super().run_inference(write_predictions)
        self._write_exp_outputs_to_azml()

   
        
def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    help_msg = 'The path to the JSON file containing the experiment definition.'
    parser.add_argument('--json-experiment', dest='json_experiment', help=help_msg, required=True)
    help_msg = 'If true, the data in the input directory is expected to be in raw netcdf format, and the ' \
               'preprocessing step to extract the data into CSV files will be run.'
    parser.add_argument('--do-preproc-extract', dest='do_preproc_extract', help=help_msg, action='store_true')
    help_msg = 'Specify whether classification output should be in a single file, or split by year or month.'
    parser.add_argument('--output-file-split', 
                        dest='output_file_split', 
                        help=help_msg, 
                        choices=xbt.common.OUTPUT_FREQS,
                        default = xbt.common.OUTPUT_SINGLE,
                        )
    help_msg = (
        'The path to the directory containing the XBT dataset in csv '
        'form, one file per year. If --preproc-path is defined, these '
        'input files will be created by the preprocessing step and '
        'written to this location.')
    parser.add_argument('--input-dir',
                        dest='input_dir',
                        help=help_msg,
                       )
    help_msg = ''
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        help=help_msg,
                       )
    parser.add_argument('--config-dir',
                        dest='config_dir',
                        help=help_msg,
                       )
    parser.add_argument('--output-root',
                        dest='output_root',
                        help=help_msg,
                       )
    return parser.parse_args()

def run_azml_experiment():
    exp_args = get_arguments(
        'Run training, inference and evaluation on a single split.'
    )
    return_code = 0
    
    
    json_desc_path = os.path.join(exp_args.config_dir, exp_args.json_experiment)
    output_dir = os.path.join(exp_args.output_root,
                              exp_args.output_dir,
                             )
    
    print(f'input directory {exp_args.input_dir}')
    print(f'output directory {exp_args.output_dir}')
    print(f'json experiment path {json_desc_path}')
    
    xbt_exp = AzureExperiment(json_descriptor=json_desc_path, 
                              data_dir=exp_args.input_dir, 
                              output_dir=output_dir,
                              do_preproc_extract=exp_args.do_preproc_extract,
                              output_split=exp_args.output_file_split,
                             )
    try:
        xbt_exp.run_single_experiment()
    except RuntimeError as e1:
        print(f'Runtime error:\n {str(e1)}')
        return_code = 1
    
    return return_code    
