import numpy 
import os
import pandas 
import unittest
from ..data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """set up mock dictionaries to be used for testing procedure"""
        self.input_dictionary_wrong = {0:{'train_dataset_name':32423,'test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 1:{'train_dataset_name':'A','test_dataset_name':32423,'useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 2:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':32423, 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 3:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':32423, 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 4:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':32423, 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 5:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':32423, 'train_split':.75, 'test_split': .25},
                                 6:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':32423, 'test_split': .25},
                                 7:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split':32423}
                                }
        self.input_dictionary_right = {'train_dataset_name':'train',
                                       'test_dataset_name':'test',
                                       'useless_features':['ID', 'town_code'],
                                       'features_to_get_labeled':['intensity','hierarchy_definition','alchool_grading'],
                                       'features_to_get_dummy':['country'], 
                                       'reshuffle':False,
                                       'train_split':.5, 
                                       'test_split': .5}
        self.train_dictionary = {'one':numpy.array([1.,2.,3.,4.]),
                                 'two':numpy.array([1.,2.,3.,numpy.nan])}
        self.test_dictionary = {'one':numpy.array([1.,2.,numpy.nan,4.]),
                                'two':numpy.array([1.,2.,3.,4.])}
      
        self.train_file_name = 'test_train.csv'
        self.test_file_name = 'test_test.csv'
        
        test_train_frame = pandas.DataFrame(self.train_dictionary)
        test_test_frame = pandas.DataFrame(self.test_dictionary)
          
        test_train_frame.to_csv(self.train_file_name,index=False)
        test_test_frame.to_csv(self.test_file_name,index=False)
        
        self.test_object = DataPreprocessor(self.train_file_name,self.test_file_name,[],[],[])

    def tearDown(self):
        os.remove(self.train_file_name)
        os.remove(self.test_file_name)
        
    def test_init(self):
        """test the correctness of the initialization"""
        
        # wrong input values: checking error raising                  
        for key in self.input_dictionary_wrong.keys():               
            init_dictionary = self.input_dictionary_wrong[key]
            self.assertRaises(ValueError,DataPreprocessor,init_dictionary['train_dataset_name'],
                                                          init_dictionary['test_dataset_name'],
                                                          init_dictionary['useless_features'],
                                                          init_dictionary['features_to_get_labeled'],
                                                          init_dictionary['features_to_get_dummy'],
                                                          init_dictionary['reshuffle'],
                                                          init_dictionary['train_split'],
                                                          init_dictionary['test_split']
                                                          )                               

        # correct input values: checking attributes values
        A = DataPreprocessor(self.input_dictionary_right['train_dataset_name'],
                             self.input_dictionary_right['test_dataset_name'],
                             self.input_dictionary_right['useless_features'],
                             self.input_dictionary_right['features_to_get_labeled'], 
                             self.input_dictionary_right['features_to_get_dummy'],
                             self.input_dictionary_right['reshuffle'],
                             self.input_dictionary_right['train_split'],
                             self.input_dictionary_right['test_split'])
        attributes_dictionary = vars(A)
        
        inputs_list = ['useless_features', 'features_to_get_labeled', 'features_to_get_dummy']
        for key in inputs_list:        
            self.assertItemsEqual(self.input_dictionary_right[key],attributes_dictionary[key],"Testing class attributes assignment")
        
        inputs_other = [element for element in attributes_dictionary.keys() if element not in inputs_list]
        for key in inputs_other:        
            self.assertEqual(self.input_dictionary_right[key],attributes_dictionary[key],"Testing class attributes assignment")
            
    def test_load_data(self):
        """test input data loading"""
                 
        self.test_object.load_data()
        self.assertItemsEqual(self.test_object.data.columns,self.train_dictionary.keys(),'Testing correct loading of data frame column keys')

        for key in self.train_dictionary.keys():
            stacked_array = numpy.hstack((self.train_dictionary[key],self.test_dictionary[key]))
            numpy.testing.assert_array_equal(stacked_array,self.test_object.data[key].values,err_msg='Testing data values for column '+key)

    def test_recode_column_values(self):
        """test recoding of column values to sliced strings"""
        
        self.test_object.load_data()
        test_frame = self.test_object.data
        
        #adding a column with long numbers
        numerical_values = numpy.array([10001,1234563,321000,44441,34567,666666,7868436,897823])
        test_frame['three'] = pandas.Series(numerical_values,index=range(4)+range(4))
        
        self.test_object.recode_column_values('three',2,5)
        sliced_column = numpy.array(['001','345','100','441','567','666','684','782'])
        
        numpy.testing.assert_equal(test_frame['three'].values,sliced_column,err_msg='Testing the recoding of column values to string and slicing')
        
    def test_remove_useless_features(self):
        """test dropping of columns according to the keys stored in the 'useless_features' attribute"""
        
        self.test_object.load_data()
        self.test_object.useless_features = ['car','bike']
        
        self.test_object.data['car']=''
        self.test_object.data['bike']=''
        self.test_object.data['truck']=''
        
        self.test_object.remove_useless_features()
        self.assertItemsEqual(self.test_object.data.columns,['one','two','truck'],'Testing correct column dropping')

    def test_categorical_features_to_label(self):
        """test conversion of categorical feature to ordinal"""
        
        self.test_object.load_data()
        self.test_object.features_to_get_labeled = ['car']

        self.test_object.data['car']=pandas.Series(['a','b','c','d','e','f','g','h'],index=range(0,4)+range(0,4))

        self.test_object.categorical_features_to_label()
        numpy.testing.assert_array_equal(self.test_object.data['car'].values,numpy.array(range(8)),err_msg='Testing correct ordinal encoding of categorical variables')
        
    def test_categorical_features_to_dummy(self):
        """test conversion of categorical features to dummy"""

        self.test_object.load_data()
        self.test_object.features_to_get_dummy = ['car']
        
        self.test_object.data['car']=pandas.Series(['a','a','a','b','b','c','d','d'],index=range(0,4)+range(0,4))
        
        self.test_object.categorical_features_to_dummy()
    
        new_labels=['car_a','car_b','car_c']
        count_arrays=[numpy.array([1,1,1,0,0,0,0,0]),
                      numpy.array([0,0,0,1,1,0,0,0]),
                      numpy.array([0,0,0,0,0,1,0,0]),
                      numpy.array([0,0,0,0,0,0,1,1])]
        for index,key in enumerate(new_labels):
            numpy.testing.assert_array_equal(count_arrays[index],self.test_object.data[key].values,err_msg='Testing correct conversion to dummies')
            
    def test_split_to_train_test(self):
        """test correct train/test splitting"""
        
        self.test_object.load_data()
        self.test_object.reshuffle = False
        self.test_object.split_to_train_test()
        
        pandas.testing.assert_frame_equal(self.test_object.data.iloc[0:6,:],self.test_object.new_train)
        pandas.testing.assert_frame_equal(self.test_object.data.iloc[6:,:],self.test_object.new_test)