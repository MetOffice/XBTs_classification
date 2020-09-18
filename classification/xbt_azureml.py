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
            
        
class AzureExperiment(classification.experiment.ClassificationExperiment):
    
    def __init__(self, json_descriptor, input_dataset_name, output_datastore_name,  output_datastore_dir, output_split, do_preproc_extract=False):
        self._azml_run = azureml.core.run.Run.get_context()
        self._azml_experiment = self._azml_run.experiment
        self._azml_workspace = self._azml_experiment.workspace
        self._azml_ds_name = input_dataset_name
        
        self._temp_output = tempfile.TemporaryDirectory()
        
        self._azml_output_datastore_name = output_datastore_name
        self._output_datastore_dir = output_datastore_dir
        
        self._azml_output_datastore = azureml.core.Datastore.get(self._azml_workspace, 
                                                            self._azml_output_datastore_name)        
        
        
        # As far as I can tell, there isn't a good way to find out
        # where AzureML has put the directory passed in as the source_directory
        # argument to the config constructor (e.g. SKLearn), which is passed to the sumbit function,
        # that is copied across to the azure compute cluster for runing. We know this files will
        # be in the classification subsdirectory of that location, so we use this hack to find
        # the source directory location. Hopefully a more elegant solution can be found.
        self._source_dir = str(pathlib.Path(__file__).absolute().parent.parent)

        
        #json experiment descriptor path is relative to the source dir
        json_exp_path_full = os.path.join(self._source_dir, json_descriptor)

        super().__init__(json_descriptor=json_exp_path_full, 
                         data_dir=None, 
                         output_dir=str(self._temp_output), 
                         output_split=output_split, 
                         do_preproc_extract=do_preproc_extract)
    
    def _write_exp_outputs_to_azml(self):
        # upload small files to the run
        if self.metrics_out_path:
            print(f'uploading metrics file {self.metrics_out_path} to AzML run')
            self._azml_run.upload_file(os.path.split(self.metrics_out_path)[1],
                                       self.metrics_out_path)
        if self.scores_out_path:
            print(f'uploading scores file {self.scores_out_path} to AzML run')
            self._azml_run.upload_file(os.path.split(self.scores_out_path)[1],
                                       self.scores_out_path)
            
        # upload predictions
        print('uploading predictions to output files:\n' + '\n'.join(self.predictions_out_path_list))
        self._azml_output_datastore.upload_files(self.predictions_out_path_list,
                                                 target_path=self._output_datastore_dir,
                                                 overwrite=True,
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

    def _construct_dataset_obj(self):
        if self._do_preproc_extract:
            self._csv_tmp_dir = tempfile.TemporaryDirectory()
            self.dataset = AzureDataset(
                year_range=self.year_range, 
                azml_ws=self._azml_workspace,
                azml_dataset_name=self._azml_ds_name,
                do_preproc_extract=self._do_preproc_extract,
                pp_prefix=self.preproc_params['prefix'],
                pp_suffix=self.preproc_params['suffix'],
                pp_csv_dir = self._csv_tmp_dir,
            )
        else:
            self.dataset = AzureDataset(
                year_range=self.year_range, 
                azml_ws=self._azml_workspace,
                azml_dataset_name=self._azml_ds_name,
            )
    
        
def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    help_msg = 'The path to the JSON file containing the experiment definition.'
    parser.add_argument('--json-experiment', dest='json_experiment', help=help_msg, required=True)
    help_msg = ('The path to the directory containing the XBT dataset in csv '
                'form, one file per year. If --preproc-path is defined, these '
                'input files will be created by the preprocessing step and '
                'written to this location.')
    parser.add_argument('--input-dataset-name', dest='input_dataset_name', help=help_msg, required=True)
    help_msg = ('If present, the input dataset should contain raw netcdf data '
                'which will be preprocessed into CSVs as part of the '
                'experiment run.')
    parser.add_argument('--do-preproc-extract', dest='do_preproc_extract', help=help_msg, action='store_true')
    help_msg = 'Specify whether classification output should be in a single file, or split by year or month.'
    parser.add_argument('--output-file-split', 
                        dest='output_file_split', 
                        help=help_msg, 
                        choices=xbt.common.OUTPUT_FREQS,
                        default = xbt.common.OUTPUT_SINGLE,
                       )
    help_msg = 'Name of the datastore to write the outputs to.'
    parser.add_argument('--output-datastore-name',
                       dest='output_datastore_name',
                       help=help_msg)
    help_msg = 'Location in the datastore to write the outputs to.'
    parser.add_argument('--output-datastore-dir',
                       dest='output_datastore_dir',
                       help=help_msg)
    return parser.parse_args()

def run_azml_experiment():
    exp_args = get_arguments(
        'Run training, inference and evaluation on a single split.'
    )
    return_code = 0
    #TODO: how can we specify that we should do the preprocessing? Maybe a tag in the experimet which specifies whether
    # the azure dataset contain the raw netcdf files or the preprocessed CSV files?

    # we need to figure out how to upload the data to an azure blob. This should probably be done through a datastore object.
    # for now just creating a temp dir to getting it running, eventually the contents should be uploaded to the blob store 
    # in the experiment
    with tempfile.TemporaryDirectory() as output_dir:
        xbt_exp = AzureExperiment(json_descriptor=exp_args.json_experiment, 
                                  input_dataset_name=exp_args.input_dataset_name, 
                                  output_datastore_name=exp_args.output_datastore_name,
                                  output_datastore_dir=exp_args.output_datastore_dir,
                                  do_preproc_extract=exp_args.do_preproc_extract,
                                  output_split=exp_args.output_file_split,
                                 )
        try:
            xbt_exp.run_single_experiment()
        except RuntimeError as e1:
            print(f'Runtime error:\n {str(e1)}')
            return_code = 1
    
    return return_code    
