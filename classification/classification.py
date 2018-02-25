"""Classification for a single experimental set up"""

import argparse
import ast
import copy
import importlib
import json
import numpy
import os
import pandas
import time
from preprocessing.data_preprocessing import  DataPreprocessor
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
pandas.options.mode.chained_assignment = None

class NumpyEncoder(json.JSONEncoder):
    """Encoder to print the result of tuning into a json file"""
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ClassificationExperiment(object):
    def __init__(self, train_file_name, test_file_name, json_descriptor):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.json_descriptor = json_descriptor
        self.primary_keys = ['learner', 'input_features', 'output_features','operations','rescale_all', 'tuning']
        self.dictionary = None
        self.output_maps = None
        self.preprocessor = None
        self.train_set = None
        self.test_set = None
        
        
    def read_json_file(self):
        """Open json descriptor file, load its content into a dictionary"""
        
        if not os.path.isfile(self.json_descriptor):
            raise ValueError('Missing json descriptor!')
        else:
            with open(self.json_descriptor) as json_data:
                dictionary = json.load(json_data)
                json_data.close()
            self.dictionary_sanity_check(dictionary)

    def dictionary_sanity_check(self, test_dictionary):
        """Check that the input dictionary contains primary keys defined when the class is instantiated"""
        
        if set(test_dictionary.keys()) != set(self.primary_keys):
            raise ValueError('Primary keys in json descriptor are wrong!')
        else:
            self.dictionary = test_dictionary
        
    def get_datasets(self):
        """Load train and test datasets, determines useless features, features to be mapped to dummy and to integers"""
        
        init_dictionary = copy.deepcopy(self.dictionary['input_features'])
        init_dictionary['train_dataset_name'] = self.train_file_name
        init_dictionary['test_dataset_name'] = self.test_file_name
        self.preprocessor = DataPreprocessor(**init_dictionary)
        self.train_set, self.test_set = self.preprocessor.load_data(stack = False)
            
    def remove_useless_features(self):
        """Remove other features we would not be interested in"""
        
        self.preprocessor.remove_useless_features(self.train_set)
        self.preprocessor.remove_useless_features(self.test_set)
                    
    def apply_operations(self, diagnostic =  False):
        """Read operations to be applied, call the necessary modules and functions. Print relevant information if diagnostic enabled"""
        
        operations = self.dictionary['operations']
        for key, operation in operations.iteritems():
            
            module_name = operation['module']
            function_name = operation['function']
            inputs = operation['inputs']

            if diagnostic:
                print('operation: '+key)
                print('module: '+ module_name)
                print('function: '+ function_name)
                print('inputs:')
                print(inputs)
  
            module = importlib.import_module(module_name)
            function = getattr(module,function_name)
            
            # Apply operation on training set
            inputs['dataframe'] = self.train_set
            function(**inputs)

            # Apply operation on training set
            inputs['dataframe'] = self.test_set
            function(**inputs)

    def get_class_codes_maps(self):
        """Help method for storing information about class codes for each output target"""
        
        temp_frame = pandas.concat([self.train_set,self.test_set])
        encoder = LabelEncoder()
        
        self.output_maps = {}
        for output_target in self.dictionary['output_features']:
            encoder.fit(temp_frame[output_target])
            encoder_name_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_,))
            self.output_maps[output_target] = encoder_name_mapping

    def imputation(self):
        """Impute categorical and numerical missing data"""
        
        numerical = self.train_set._get_numeric_data().columns
        categorical = [column for column in self.train_set.columns if column not in numerical]
        
        self.train_set, self.test_set = self.preprocessor.impute_numerical_nans(self.train_set, self.test_set, numerical, 'median', 0)
        self.train_set, self.test_set = self.preprocessor.impute_categorical_nans(self.train_set, self.test_set, categorical, 'local', 0)
 
    def categorical_features_conversion(self):
        """Convert categorical features into ordinakl integers or dummy variables, depending on the information conatined in the json descriptor"""
        
        index = self.train_set.shape[0]
        temp_frame = pandas.concat([self.train_set,self.test_set])
        
        if self.preprocessor.features_to_get_labeled:
            self.preprocessor.categorical_features_to_label(temp_frame)
        if self.preprocessor.features_to_get_dummy:
            self.preprocessor.categorical_features_to_dummy(temp_frame)

        self.train_set = temp_frame.iloc[0:index,:]
        self.test_set = temp_frame.iloc[index:,:]
        
    def rescale(self):
        """Rescale the collection of input features described by the user"""
        if self.dictionary['rescale_all']:
            # rescale all input features
            self.preprocessor.features_to_rescale = [column for column in self.train_set.columns if column not in self.dictionary['output_features']]
            self.preprocessor.rescale_features(self.train_set, self.test_set)
        else:
            self.preprocessor.rescale_features(self.train_set, self.test_set)
            
    def generate_train_test_arrays(self):
        """Generate input train and test arrays, and a collection of output train and test arrays"""
        
        input_features = [column for column in self.train_set.columns if column not in self.dictionary['output_features']]
        self.X_train = self.train_set[input_features].values 
        self.X_test = self.test_set[input_features].values
        
        self.y_trains = [self.train_set[column].values for column in self.dictionary['output_features']]
        self.y_tests = [self.test_set[column].values for column in self.dictionary['output_features']]

    def optimize_and_predict(self, out_path='.', year=''):
        """Get information about the learner to be used, the hyperparameters to be tuned, and the use grid-search"""
        
        learner_dictionary = self.dictionary['learner']
        learner_module_name = learner_dictionary['module_name']
        learner_class_name =  learner_dictionary['python_class']
        
        learner_module = importlib.import_module(learner_module_name)
        learner_class = getattr(learner_module, learner_class_name)
        learner = learner_class()
        
        grid_search_parameters = self.dictionary['tuning']
        grid_search_parameters['estimator'] = learner

        for key, item in grid_search_parameters['param_grid'].iteritems():
            if isinstance(item, unicode):
                grid_search_parameters['param_grid'][key] = ast.literal_eval(item)
        print('Initializing grid search')
        grid_search = GridSearchCV(**grid_search_parameters)
       
        classification_result = {}
        
        sub_directory_name = os.path.basename(self.json_descriptor).split('.')[0]
        
        print('Starting tuning procedure')
        for y_train, y_test, output_target in zip(self.y_trains, self.y_tests, self.dictionary['output_features']):
          
            start = time.time()
            grid_search.fit(self.X_train, y_train)
            end = time.time()

            print('Tuning procedure for '+ output_target+' completed, time elapsed = '+str(end-start)+' seconds.')
            prediction_probabilities = grid_search.predict_proba(self.X_test)
            prediction = grid_search.predict(self.X_test)
            classification_accuracy_score = accuracy_score(prediction, y_test)
            classification_recall_score = recall_score(prediction, y_test, average='weighted')            
            
            classification_result['probabilities'] = prediction_probabilities.mean(axis=0)
            classification_result['class_mapping'] = self.output_maps[output_target]
            classification_result['accuracy'] = classification_accuracy_score
            classification_result['recall'] = classification_recall_score
            classification_result['rescale_all'] = self.dictionary['rescale_all']
            classification_result['input_features'] = self.dictionary['input_features']
            classification_result['applied_operations'] = self.dictionary['operations'].keys()
            classification_result['best_hyperparameters'] = grid_search.best_params_
             
            self.generate_results(output_target, out_path, learner_class_name, sub_directory_name, grid_search.cv_results_, classification_result, year)
             
                        
    def generate_results(self, output_target, out_path, learner_class_name, sub_directory_name, tuning_result, classification_result, year):
        """Store results from tuning and prediction into json files, generating a proper tree of directories"""
        
        base_dir = os.path.join(out_path, learner_class_name)
        out_dir = os.path.join(base_dir,sub_directory_name)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        tuning_file_name = os.path.join(out_dir,year+'_'+output_target+'_tuning.json')
        prediction_file_name = os.path.join(out_dir,year+'_'+output_target+'_prediction.json')
        
        for name, data in zip([tuning_file_name, prediction_file_name], [tuning_result, classification_result]):
            with open(name, 'w') as fp:
                json.dump(data, fp, cls=NumpyEncoder)
    
def run_experiment(data_path, result_path, year, train_set_name, test_set_name, json_descriptor):
    """Run a single classification experiment"""
    
    train_file_name = os.path.join(data_path, train_set_name)
    test_file_name = os.path.join(data_path, test_set_name)
    
    experiment = ClassificationExperiment(train_file_name, test_file_name, json_descriptor)
    experiment.read_json_file()
    experiment.get_datasets()
    experiment.remove_useless_features()
    experiment.apply_operations()   
    experiment.get_class_codes_maps()
    experiment.imputation()
    experiment.categorical_features_conversion()
    experiment.rescale()
    experiment.generate_train_test_arrays()
    experiment.optimize_and_predict(result_path, year)
    
def main():
    """For running a specific classification experiment from shell"""
    
    parser = argparse.ArgumentParser(description='Run a specific classification experiment')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='outputpath for results')
    parser.add_argument('--year', default='',help='year')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('json_descriptor',help='json descriptor file name')
    
    args = parser.parse_args()

    run_experiment(args.path, args.outpath, args.year, args.train, args.test, args.json_descriptor)

    
if __name__ == "__main__":
    # execute only if run as a script
    main()