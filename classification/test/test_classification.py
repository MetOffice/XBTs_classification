import json
import os
import numpy 
import pandas 
import unittest
from ..classification import ClassificationExperiment
pandas.options.mode.chained_assignment = None

class TestClassificationExperiment(unittest.TestCase):
    
    def setUp(self):
        """set up mock input files and data frames to be used for testing procedure"""
    
        self.dictionary = {'input_features':{'features_to_get_labeled':['cars'],
                                             'features_to_get_dummy':['animals'],
                                             'features_to_rescale':['one','two']},
                           'operations':{'set_2_to_1':{'module':'classification.test.dummy_module',
                                                       'function':'change_value',
                                                       'inputs':{'column':'one',
                                                                 'operator':'==',
                                                                 'target_value':'2.',
                                                                 'new_value':1.}},
                                         'add_animal_cars_column':{'module':'classification.test.dummy_module',
                                                                   'function':'new_feature_creation',
                                                                   'inputs':{'left_column':'cars',
                                                                             'right_column':'animals',
                                                                             'filling_value':'gromobo'}}},
                           'rescaling':True,
                           'learner':{},
                           'tuning':{}}
    
        self.wrong_dictionary = {'input_features':{},'operations':{}, 'rescaling':{},'learner':{},'turning':{}}
       
        self.json_descriptor = 'descriptor.json'
        self.wrong_json_descriptor = 'wrong_descriptor.json'

        train_dictionary = {'one':numpy.array([2., 2., 2., numpy.nan, numpy.nan, 3., 7.]),
                            'two':numpy.array([1., 2., 3., 4., 3., 1., numpy.nan])}
        self.train_frame = pandas.DataFrame(train_dictionary)
        self.train_frame['animals'] = pandas.Series(['bird','bird','bird','bird', numpy.nan,'bird','bird'])
        self.train_frame['cars'] =  pandas.Series(['audi','audi','audi','audi', numpy.nan,'audi','audi'])
        
        test_dictionary = {'one':numpy.array([2., 2., numpy.nan, 4., 3., 3., 7.]),
                           'two':numpy.array([1., 2., 3., 4., 3., numpy.nan, numpy.nan])}
        self.test_frame = pandas.DataFrame(test_dictionary)
        self.test_frame['animals'] = pandas.Series([numpy.nan,'bird','bird','bird', 'bird','bird','bird'])
        self.test_frame['cars'] =  pandas.Series(['audi','audi','audi',numpy.nan, numpy.nan,'audi','audi'])
        
        self.train_file_name = 'test_train.csv'
        self.test_file_name = 'test_test.csv'
        
        self.train_frame.to_csv(self.train_file_name,index=False)
        self.test_frame.to_csv(self.test_file_name,index=False)

        for name, data in zip([self.json_descriptor,self.wrong_json_descriptor ],[self.dictionary, self.wrong_dictionary]):
            with open(name, 'w') as fp:
                json.dump(data, fp)

    def tearDown(self):
        os.remove(self.wrong_json_descriptor)
        os.remove(self.json_descriptor)
        os.remove(self.train_file_name)
        os.remove(self.test_file_name)
        
    def test_init(self):
        """Test correctness of the initialization"""

        experiment = ClassificationExperiment('baba', 'buba', 'biba')
        self.assertEqual(experiment.train_file_name, 'baba')
        self.assertEqual(experiment.test_file_name, 'buba')
        self.assertEqual(experiment.json_descriptor, 'biba')
        self.assertEqual(experiment.dictionary, None)
        self.assertEqual(experiment.preprocessor, None)
        self.assertEqual(experiment.train_set, None)
        self.assertEqual(experiment.test_set, None)

    def test_read_json_file_and_dictionary_sanity_check(self):
        """Test json file reader and loaded dictionary"""
        
        experiment = ClassificationExperiment('baba', 'buba', 'biba')
        self.assertRaises(ValueError,experiment.read_json_file)
        
        experiment = ClassificationExperiment('baba', 'buba', self.wrong_json_descriptor)
        self.assertRaises(ValueError,experiment.read_json_file)
        
        experiment = ClassificationExperiment('baba', 'buba', self.json_descriptor)
        experiment.read_json_file()
        self.assertDictEqual(self.dictionary, experiment.dictionary)
        
    def test_get_datasets(self):
        """Test correcto datasets loading"""
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        pandas.testing.assert_frame_equal(self.test_frame,experiment.test_set)
        pandas.testing.assert_frame_equal(self.train_frame,experiment.train_set)
        
    def test_apply_operation(self):
        """Test correct implementation of operations contained in json descriptor"""
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        experiment.apply_operations()   
        
        modified_column = 'one'
        numpy.testing.assert_array_equal(experiment.train_set[modified_column].values,numpy.array([1., 1., 1., numpy.nan, numpy.nan, 3., 7.]), err_msg='Testing correct value change for train frame')
        numpy.testing.assert_array_equal(experiment.test_set[modified_column].values,numpy.array([1., 1., numpy.nan, 4., 3., 3., 7.]), err_msg='Testing correct value change for test frame')
        
        new_label = 'cars_animals'
        self.assertTrue(new_label in experiment.train_set.columns)
        self.assertTrue(new_label in experiment.test_set.columns)
        numpy.testing.assert_array_equal(experiment.train_set[new_label].values, 'gromobo', err_msg='Testing column addition on train frame')
        numpy.testing.assert_array_equal(experiment.test_set[new_label].values, 'gromobo', err_msg='Testing column addition on train frame')
        
    def test_imputation(self):
        """Test correct imputation on train and test set"""
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        
        experiment.imputation()
        
        keys = ['one', 'two', 'animals', 'cars']
        check_values = [[numpy.array([2., 2., 2., 2., 2., 3., 7.]), numpy.array([2., 2., 2.0, 4., 3., 3., 7.])],
                        [numpy.array([1., 2., 3., 4., 3., 1., 2.5]), numpy.array([1., 2., 3., 4., 3., 2.5, 2.5])],
                        ['bird','bird'],
                        ['audi', 'audi']]
        
        for key, values in zip(keys, check_values):
            numpy.testing.assert_array_equal(experiment.train_set[key], values[0], err_msg='Checking correct data imputation for train set')
            numpy.testing.assert_array_equal(experiment.test_set[key], values[1], err_msg='Checking correct data imputation for test set')
    
    def test_categorical_features_conversion(self):
        """Test correct conversion of categorical features into numeric ones"""
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        # does need data with no nans
        experiment.imputation()
        experiment.categorical_features_conversion()
        new_columns = ['one','two','cars','animals_bird']
        
        self.assertItemsEqual(experiment.train_set.columns,new_columns)
        self.assertItemsEqual(experiment.test_set.columns,new_columns)
        numpy.testing.assert_array_equal(experiment.train_set['cars'].values, 0., err_msg='Testing correct conversion of "car" feature of train set into ordinal')
        numpy.testing.assert_array_equal(experiment.test_set['cars'].values, 0., err_msg='Testing correct conversion of "car" feature of test set into ordinal')
        numpy.testing.assert_array_equal(experiment.train_set['animals_bird'].values, 1, err_msg='Testing correct conversion of "animals" feature of train set into dummy')
        numpy.testing.assert_array_equal(experiment.test_set['animals_bird'].values, 1, err_msg='Testing correct conversion of "animals" feature of test set into dummy')
        
    def test_rescale(self):
        """Testing correctness of rescaling procedure"""
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        # does need data with no nans
        experiment.imputation()
        experiment.categorical_features_conversion()
        experiment.rescale()
        
        keys = ['one', 'two']
        check_values = [[numpy.array([-0.49656353, -0.49656353, -0.49656353, -0.49656353, -0.49656353,0.08276059,  2.40005708]), numpy.array([-1.32379273, -0.34836651,  0.62705971,  1.60248593,  0.62705971, -1.32379273,  0.1393466 ])],
        [numpy.array([-1.32379273, -0.34836651,  0.62705971,  1.60248593,  0.62705971,-1.32379273,  0.1393466 ]), numpy.array([-1.82073295, -0.66208471,  0.49656353,  1.65521178,  0.49656353, -0.08276059, -0.08276059])]]
        
        for key, values in zip(keys, check_values):
            numpy.testing.assert_array_almost_equal(experiment.train_set[key], values[0], err_msg='Checking correct data rescaling for train set')
            numpy.testing.assert_array_almost_equal(experiment.test_set[key], values[1], err_msg='Checking correct data rescaling for test set')
