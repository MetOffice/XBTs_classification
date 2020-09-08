import azureml.core 
import azureml.core.run
import dataexploration.xbt_dataset
import classification.experiment

class AzureDataset(dataexploration.xbt_dataset.XbtDataset):
    def __init__(self, year_range, azml_ws, azml_dataset_name, 
                 df=None, use_dask=False, load_profiles=False, load_quality_flags=False, nc_dir=None, pp_prefix='', pp_suffix=''):
        
        # Set up the azure dataset and mount the files in a temporary directory. This location
        # is then passed to the parent class which can proceed with loading data in the normal
        #
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
                            nc_dir,
                            pp_prefix,
                            pp_suffix,
                      )
        
class AzureExperiment(classification.experiment.ClassificationExperiment):
    
    def __init__(self, json_descriptor, ds_name, output_dir, output_split, preproc_dir):
        self._azml_run = azureml.core.run.Run.get_context()
        self._azml_experiment = self._azml_run.experiment
        self._azml_workspace = self._azml_experiment.workspace
        
        # get json descriptor, from a blob in a datastore or from project directory
        # we may need some experimentation to find where we are since this data is 
        # not supplied by the azureML API as far as I can find
               
        super().__init__(json_descriptor, 
                         data_dir, 
                         output_dir, 
                         output_split, 
                         preproc_dir)
    
    def read_json_file(self):
        # resolve absolute path to JSON experiment file. This should be in the project directory
        super().read_json_file()
        
    def load_dataset(self):
        # create an azure dataset rather than the base XbtDataset (use factory method?)
        # pass in additional azure stuff, like a workspace and an experiment
        # this may require another overloaded funcation, like _construct_datasett, called from loaddataset, 
        # which create an AzureDataset object rather than XbtDataset. We can't do that from an overload
        # load_dataset and then call then base function, because the base function just creates the base
        # object, which we don't want to change. Instead have the base load_dataset call the _construct_dataset
        # function. The base implementation constructs an XbtDataset object, but our AzureExperiment will overload
        # _construct_dataset which will create an AzureDataset object, which handles the mounting.
        super().load_dataset()
        
def get_arguments():
    parser = argparse.ArgumentParser(description=description)
    help_msg = 'The path to the JSON file containing the experiment definition.'
    parser.add_argument('--json-experiment', dest='json_experiment', help=help_msg, required=True)
    help_msg = ('The path to the directory containing the XBT dataset in csv '
                'form, one file per year. If --preproc-path is defined, these '
                'input files will be created by the preprocessing step and '
                'written to this location.')
    parser.add_argument('--input-dataset-name', dest='input_dataset_name', help=help_msg, required=True)
    help_msg = 'The path to the directory containing files to be preprocessed, typically netCDF files from WOD or similar source.'
    parser.add_argument('--output-path', dest='output_path', help=help_msg, required=True)
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

    xbt_exp = AzureExperiment(exp_args.json_experiment, 
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
