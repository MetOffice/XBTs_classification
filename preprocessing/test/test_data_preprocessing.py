import numpy 
import os
import pandas 
import unittest
from ..data_preprocessing import DataPreprocessor
pandas.options.mode.chained_assignment = None

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """set up mock dictionaries and data frames to be used for testing procedure"""
        
        self.input_dictionary_wrong = {0:{'train_dataset_name':32423,'test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 1:{'train_dataset_name':'A','test_dataset_name':32423,'useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'features_to_rescale':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 2:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':32423, 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'features_to_rescale':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 3:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':32423, 'features_to_get_dummy':[], 'features_to_rescale':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 4:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':32423, 'features_to_rescale':[], 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 5:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy': [], 'features_to_rescale':32423, 'reshuffle':True, 'train_split':.75, 'test_split': .25},
                                 6:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':32423, 'train_split':.75, 'test_split': .25},
                                 7:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':32423, 'test_split': .25},
                                 8:{'train_dataset_name':'A','test_dataset_name':'A','useless_features':[], 'features_to_get_labeled':[], 'features_to_get_dummy':[], 'reshuffle':True, 'train_split':.75, 'test_split':32423}
                                }
        self.input_dictionary_right = {'train_dataset_name':'train',
                                       'test_dataset_name':'test',
                                       'useless_features':['ID', 'town_code'],
                                       'features_to_get_labeled':['intensity','hierarchy_definition','alchool_grading'],
                                       'features_to_get_dummy':['country'],
                                       'features_to_rescale':['price'],
                                       'reshuffle':False,
                                       'train_split':.5, 
                                       'test_split': .5}
                                       
        self.train_dictionary = {'one':numpy.array([1.,2.,3.,4.]),
                                 'two':numpy.array([1.,2.,3.,numpy.nan])}
        self.test_dictionary = {'one':numpy.array([1.,2.,numpy.nan,4.]),
                                'two':numpy.array([1.,2.,3.,4.])}
      
        self.train_file_name = 'test_train.csv'
        self.test_file_name = 'test_test.csv'
        
        self.test_train_frame = pandas.DataFrame(self.train_dictionary)
        self.test_test_frame = pandas.DataFrame(self.test_dictionary)

        self.test_train_frame.to_csv(self.train_file_name,index=False)
        self.test_test_frame.to_csv(self.test_file_name,index=False)
        
    def tearDown(self):
        os.remove(self.train_file_name)
        os.remove(self.test_file_name)
        
    def test_init(self):
        """test the correctness of the initialization"""
        
        # wrong input values: checking error raising                  
        for key in self.input_dictionary_wrong.keys():               
            init_dictionary = self.input_dictionary_wrong[key]
            self.assertRaises(ValueError,DataPreprocessor,**init_dictionary)
   
        # correct input values: checking attributes values
        A = DataPreprocessor(**self.input_dictionary_right)
        attributes_dictionary = vars(A)
        
        inputs_list = ['useless_features', 'features_to_get_labeled', 'features_to_get_dummy', 'features_to_rescale']
        for key in inputs_list:        
            self.assertItemsEqual(self.input_dictionary_right[key],attributes_dictionary[key],"Testing class attributes assignment")
        
        inputs_other = [element for element in attributes_dictionary.keys() if element not in inputs_list]
        for key in inputs_other:        
            self.assertEqual(self.input_dictionary_right[key],attributes_dictionary[key],"Testing class attributes assignment")
            
    def test_load_data(self):
        """test input data loading and data frame stacking"""
                 
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name)
        test_train_frame, test_test_frame = test_preprocessor.load_data()
        
        pandas.testing.assert_frame_equal(test_train_frame,self.test_train_frame)
        pandas.testing.assert_frame_equal(test_test_frame,self.test_test_frame)
        
        stack = True
        
        test_data_frame = test_preprocessor.load_data(stack)
        pandas.testing.assert_frame_equal(test_data_frame,pandas.concat([self.test_train_frame,self.test_test_frame]))
        
    def test_recode_column_values(self):
        """test recoding of column values to sliced strings"""
                
        numerical_values = {'numbers':numpy.array([10001,1234563,321000,44441,34567,666666,7868436,897823])}
        test_frame = pandas.DataFrame(numerical_values)
        
        DataPreprocessor.recode_column_values(test_frame,'numbers',2,5)
        sliced_column = numpy.array(['001','345','100','441','567','666','684','782'])
        
        numpy.testing.assert_equal(test_frame['numbers'].values,sliced_column,err_msg='Testing the recoding of column values to string and slicing')
        
    def test_remove_useless_features(self):
        """test dropping of columns according to the keys stored in the 'useless_features' attribute"""
      
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name,useless_features=['car','bike'])
        
        self.test_train_frame['car']=''
        self.test_train_frame['bike']=''
        test_preprocessor.remove_useless_features(self.test_train_frame)
        self.assertItemsEqual(self.test_train_frame.columns,['one','two'],'Testing correct column dropping')

    def test_categorical_features_to_label(self):
        """test conversion of categorical feature to ordinal"""
        
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name,features_to_get_labeled=['car'])
 
        self.test_train_frame['car']=pandas.Series(['a','b','c','d'])

        test_preprocessor.categorical_features_to_label(self.test_train_frame)
        numpy.testing.assert_array_equal(self.test_train_frame['car'].values,numpy.array(range(4)),err_msg='Testing correct ordinal encoding of categorical variables')
        
    def test_categorical_features_to_dummy(self):
        """test conversion of categorical features to dummy"""

        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name,features_to_get_dummy=['car'])
        
        self.test_train_frame['car']=pandas.Series(['a','a','b','c'])
        test_preprocessor.categorical_features_to_dummy(self.test_train_frame)
        new_labels=['car_a','car_b','car_c']
        count_arrays=[numpy.array([1,1,0,0]),
                      numpy.array([0,0,1,0]),
                      numpy.array([0,0,0,1])]
        for index,key in enumerate(new_labels):
            numpy.testing.assert_array_equal(count_arrays[index],self.test_train_frame[key].values,err_msg='Testing correct conversion to dummies')
            
        self.test_train_frame.drop(labels=new_labels,axis=1,inplace = True)
        
    def test_impute_numerical_nans(self):
        """test nan values imputing"""
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name)

        # return arrays, imputing over row index
        train_array, test_array = test_preprocessor.impute_numerical_nans(self.test_train_frame,self.test_test_frame, columns = ['one','two'], strategy='mean', axis=0,as_frame = False)        
        train_expected = numpy.array([[1.,1.],[2.,2.],[3.,3.],[4.,2.]])
        test_expected = numpy.array([[1.,1.],[2.,2.],[2.5,3.],[4.,4.]])
        numpy.testing.assert_array_equal(train_array,train_expected,err_msg='Testing correct imputation procedure of train array: using mean')
        numpy.testing.assert_array_equal(test_array,test_expected,err_msg='Testing correct imputation procedure of test array: using mean')

        # return frames, imputing over columns index
        self.test_train_frame['two'].iloc[3]=numpy.nan
        self.test_test_frame['one'].iloc[2]=numpy.nan
        train_frame, test_frame = test_preprocessor.impute_numerical_nans(self.test_train_frame,self.test_test_frame, columns = ['one','two'], strategy='mean', axis=1,as_frame = True)
        train_expected = numpy.array([[1.,1.],[2.,2.],[3.,3.],[4.,4.]])
        test_expected = numpy.array([[1.,1.],[2.,2.],[3.,3.],[4.,4.]])        
        pandas.testing.assert_frame_equal(train_frame,pandas.DataFrame(train_expected,columns=self.test_train_frame.columns))
        pandas.testing.assert_frame_equal(test_frame,pandas.DataFrame(test_expected,columns=self.test_test_frame.columns))

    def test_impute_categorical_nans(self):
        """test nan categorical imputing"""
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name)

        train_animals = pandas.Categorical(['bird','bird','bird','bird', numpy.NaN,'bird','bird'], categories=["bird", "cat", "dog","cheetah"])
        train_cars = pandas.Categorical(['audi','audi','audi','audi', numpy.NaN,'audi','audi'], categories=["audi", "fiat", "opel", "volvo"])
        train_frame = pandas.DataFrame({'cars':train_cars,'animals':train_animals})
        
        test_animals = pandas.Categorical([numpy.NaN,'bird','bird','bird','bird','bird','bird'], categories=["bird", "cat", "dog","cheetah"])
        test_cars = pandas.Categorical(['audi','audi','audi',numpy.NaN, numpy.NaN,'audi','audi'], categories=["audi", "fiat", "opel", "volvo"])
        test_frame = pandas.DataFrame({'cars':test_cars,'animals':test_animals})
        
        # check on criterion value
        self.assertRaises(ValueError,test_preprocessor.impute_categorical_nans,train_frame,test_frame,['animals'],'bafo',0,False)
        # return arrays
        imputed_train_array, imputed_test_array = test_preprocessor.impute_categorical_nans(train_frame, test_frame, ['animals'], strategy='local', axis=0, as_frame = False)
        expected_result = numpy.array(['bird','bird','bird','bird','bird','bird','bird'])
        numpy.testing.assert_array_equal(imputed_train_array[:,0],expected_result,err_msg = 'Testing imputation of train set, local strategy')
        numpy.testing.assert_array_equal(imputed_test_array[:,0],expected_result,err_msg = 'Testing imputation of test set, local strategy')
      
        # return frames
        imputed_train_frame, imputed_test_frame = test_preprocessor.impute_categorical_nans(train_frame, test_frame, ['cars'], strategy='global', axis=0, as_frame = True) 
        expected_animals = pandas.Categorical(['bird','bird','bird','bird', 'bird','bird','bird'], categories=["bird", "cat", "dog","cheetah"])
        expected_cars = pandas.Categorical(['audi','audi','audi','audi', 'audi','audi','audi'], categories=["audi", "fiat", "opel", "volvo"])
        expected_frame = pandas.DataFrame({'cars':expected_cars,'animals':expected_animals})
        pandas.testing.assert_frame_equal(expected_frame, imputed_train_frame)
        pandas.testing.assert_frame_equal(expected_frame, imputed_test_frame)        
        
    def test_rescale_features(self):
        """test features rescaling"""
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name,features_to_rescale=['babo'])
        
        train_frame = pandas.DataFrame(numpy.array([[1.,2.],[1.,2.],[.1,.3],[4.,.2]]) ,columns=['babo','bibo'])
        test_frame = pandas.DataFrame(numpy.array([[5.,2.],[1.,2.],[1.,.3],[4.,.2]]),columns=['babo','bibo'])
        
        train_array, test_array = test_preprocessor.rescale_features(train_frame, test_frame, as_frame = False)
        train_expected = numpy.array([[-0.35583   ,  2.],[-0.35583   ,  2.],[-0.96582428, .3],[ 1.67748427, .2]])
        test_expected = numpy.array([[ 2.3552557 ,  2.],[-0.35583   ,  2.],[-0.35583   , .3 ],[ 1.67748427, .2]])
        numpy.testing.assert_array_almost_equal(train_array,train_expected,err_msg='Testing correct rescaling procedure of train array')
        numpy.testing.assert_array_almost_equal(test_array,test_expected,err_msg='Testing correct rescaling procedure of test array')
    
    def test_split_to_train_test(self):
        """test correct train/test splitting"""
        
        test_preprocessor = DataPreprocessor(self.train_file_name, self.test_file_name,reshuffle=False)
        test_data = test_preprocessor.load_data(True)
        
        new_train_frame, new_test_frame = test_preprocessor.split_to_train_test(test_data)
        
        pandas.testing.assert_frame_equal(test_data.iloc[0:6,:],new_train_frame)
        pandas.testing.assert_frame_equal(test_data.iloc[6:,:],new_test_frame)