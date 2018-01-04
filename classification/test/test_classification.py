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
    
        self.dictionary = {'input_features':{},'operations':{}, 'rescaling':{},'learner':{},'tuning':{}}
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
        
        experiment = ClassificationExperiment(self.train_file_name, self.test_file_name, self.json_descriptor)
        experiment.read_json_file()
        experiment.get_datasets()
        pandas.testing.assert_frame_equal(self.test_frame,experiment.test_set)
        pandas.testing.assert_frame_equal(self.train_frame,experiment.train_set)
        
        