"""Classification for a single experimental set up"""

import importlib
import json
import os
import pandas
from preprocessing.data_preprocessing import  DataPreprocessor

class ClassificationExperiment(object):
    def __init__(self, train_file_name, test_file_name, json_descriptor):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.json_descriptor = json_descriptor
        self.primary_keys = ['learner', 'input_features', 'operations', 'rescaling', 'tuning']
        self.dictionary = None
        self.preprocessor = None
        self.train_set = None
        self.test_set = None
        
    def read_json_file(self):
        """Open json descriptor file, load its content into a dictionary"""
        
        if not os.path.isfile(self.json_descriptor):
            raise ValueError('Missing Json descriptor!')
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
        """Load train and test datasets, determines features to be mapped to dummy and to integers"""
        
        init_dictionary = self.dictionary['input_features']
        init_dictionary['train_dataset_name'] = self.train_file_name
        init_dictionary['test_dataset_name'] = self.test_file_name
        self.preprocessor = DataPreprocessor(**init_dictionary)
        self.train_set, self.test_set = self.preprocessor.load_data(stack = False)
            
    def apply_operations(self, diagnostic =  False):
        """Read operations to be applied, call the necessary modules and functions. Print relevant information if diagnostic enabled"""
        
        operations = self.dictionary['operations']
        for key, operation in operations.iteritems():
            
            module_name = operation['module']
            function_name = operation['function']
            inputs = operation['inputs']

            if diagnostic:
                print('operation: '+key)
                print('moudule: '+ module_name)
                print('moudule: '+ function_name)
                print('inputs: '+inputs)
  
            module = importlib.import_module()
            function = module.get_attr(function_name)
            
            # Apply operation on training set
            inputs['dataframe'] = self.train_set
            function(**inputs)

            # Apply operation on training set
            inputs['dataframe'] = self.test_set
            function(**inputs)

    def imputation(self):
        """Impute categorical and numerical missing data"""
        
        numerical = self.train_set._get_numeric_data()
        categorical = [column for column in self.train.columns if column not in numerical]
        
        self.train_set, self.test_set = self.preprocessor.impute_numerical_nans(self.train_set, self.test_set, numerical, 'median', 0)
        self.train_set, self.test_set = self.preprocessor.impute_categorical_nans(self.train_set, self.test_set, categorical, 'local', 0)
 
    def categorical_features_conversion(self):
        """Convert categorical features into ordinakl integers or dummy variables, depending on the information conatined in the json descriptor"""
        
        index = self.train_set.shape[0]
        temp_frame = pandas.concat([self.train_set,self.test_set])
        
        self.preprocessor.categorical_features_to_label(temp_frame)
        self.preprocessor.categorical_features_to_dummy(temp_frame)

        self.train_set = temp_frame.iloc[0:index,:]
        self.test_set = temp_frame.iloc[index:,:]

        self.preprocessor.features_to_rescale = self.train_set.columns
        
    def rescale(self):
        if self.dictionary['rescaling']:
            self.preprocessor.rescale_features(self.train_set, self.test_set)
    
   #*=============================================
   
    # call tuning module
    # return tuned parameter
    # if flag, return tuning procedure
    
    # call prediction
    # return accuracy, recall, and averaged probabilities along the sample