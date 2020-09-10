import pathlib

import azureml.core 
import azureml.core.run
import dataexploration.xbt_dataset
import classification.experiment

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
    
    def __init__(self, json_descriptor, ds_name, output_dir, output_split, do_preproc_extract=False):
        self._azml_run = azureml.core.run.Run.get_context()
        self._azml_experiment = self._azml_run.experiment
        self._azml_workspace = self._azml_experiment.workspace
        self._azml_ds_name = ds_name
        
        if output_dir is None:
            self._temp_output = tempfile.TemporaryDirectory()
            output_dir = self._temp_output
        # As far as I can tell, there isn't a good way to find out
        # where AzureML has put the directory passed in as the source_directory
        # argument to the config constructor (e.g. SKLearn), which is passed to the sumbit function,
        # that is copied across to the azure compute cluster for runing. We know this files will
        # be in the classification subsdirectory of that location, so we use this hack to find
        # the source directory location. Hopefully a more elegant solution can be found.
        self._source_dir = str(pathlib.Path(__file__).absolute().parent.parent)

        
        #json experiment descriptor path is relative to the source dir
        json_exp_path_full = os.path.join(self._source_dir, json_descriptor)
               
        super().__init__(json_exp_path_full, 
                         None, 
                         output_dir, 
                         output_split, 
                         preproc_dir)
        
    
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
    
        
def get_arguments():
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
    return parser.parse_args()

def run_azml_experiment():
    exp_args = get_arguments(
        'Run training, inference and evaluation on a single split.'
    )
    return_code = 0
    #TODO: how can we specify that we should do the preprocessing? Maybe a tag in the experimet which specifies whether
    # the azure dataset contain the raw netcdf files or the preprocessed CSV files?

    json_descriptor, ds_name, output_dir, output_split, do_preproc_extract=False
    
    # we need to figure out how to upload the data to an azure blob. This should probably be done through a datastore object.
    # for now just creating a temp dir to getting it running, eventually the contents should be uploaded to the blob store 
    # in the experiment
    with tempfile.TemporaryDirectory() as output_dir:
        xbt_exp = AzureExperiment(json_descriptor=exp_args.json_experiment, 
                                  ds_name=exp_args.input_dataset_name, 
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
