import jsons
import numpy 
import pandas 
import unittest
from ..features_engineering import probe_type_output
from ..features_engineering import map_attribute_value_to_NaN
from ..features_engineering import  convert_list_to_integer
pandas.options.mode.chained_assignment = None

class TestFeaturesEngineering(unittest.TestCase):
    
    def setUp(self):
        
        test_dictionary = {'instrument':numpy.array(['STD: 9040 STD (UNKNOWN BRAND)', 
                                                     'STD: 9060 STD (UNKNOWN BRAND)',
                                                     'XBT: AXBT (TSK - TSURUMI SEIKI Co.)',
                                                     'XBT: AXBT (UNKNOWN BRAND AND TYPE)', 
                                                     'XBT: AXBT 536 (SPARTON)',
                                                     'XBT: DEEP BLUE (SIPPICAN)',
                                                     'XBT: DEEP BLUE (TSK - TSURUMI SEIKI Co.)',
                                                     'XBT: DEEP BLUE, UNKNOWN BRAND', 
                                                     'XBT: FAST DEEP (SIPPICAN)',
                                                     'XBT: FAST DEEP, UNKNOWN BRAND',
                                                     'XBT: SUBMARINE-LAUNCHED EXPENDABLE BATHYTHERMOGRAPH (SSXBT) (SIPPICAN)',
                                                     'XBT: T10 (SIPPICAN)', 
                                                     'XBT: T10 (TSK - TSURUMI SEIKI Co.)',
                                                     'XBT: T10 (UNKNOWN BRAND)', 
                                                     'XBT: T11 (SIPPICAN)',
                                                     'XBT: T11 (UNKNOWN BRAND)', 
                                                     'XBT: T4 (SIPPICAN)',
                                                     'XBT: T4 (TSK - TSURUMI SEIKI Co.)', 
                                                     'XBT: T4 (UNKNOWN BRAND)']),
                           'max_depth':numpy.array([-1,-2,.4,10,1.,2.,3.,4.,-4,5,6.,9.,3,-45,-.0002,2,3,10,-3]),
                           'depth_profile':numpy.array(['[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01]',
                                                        '[ -1.72580719e-01   1.88086176e+00]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]',
                                                        '[]',
                                                        '[]',
                                                        '[ -1.72580719e-01   1.88086176e+00   2.90710282e+00   9.06809235e+00\n]']),
                           'platform':numpy.array(['a', 'O', 'b', 'c', 'O', 'd', 'O', 'e', 'f', 'O', 'g', 'h', 'O', 'i', 'j', 'k', 'l', 'm', 'n'])}

        result_dictionary = {'instrument_type_and_manifacturer':numpy.array(['9040 STD,UNKNOWN BRAND', 
                                                                             '9060 STD,UNKNOWN BRAND',
                                                                             'AXBT,TSK - TSURUMI SEIKI Co.',
                                                                             'AXBT,UNKNOWN BRAND AND TYPE', 
                                                                             'AXBT 536,SPARTON',
                                                                             'DEEP BLUE,SIPPICAN',
                                                                             'DEEP BLUE,TSK - TSURUMI SEIKI Co.',
                                                                             'DEEP BLUE,UNKNOWN BRAND', 
                                                                             'FAST DEEP,SIPPICAN',
                                                                             'FAST DEEP,UNKNOWN BRAND',
                                                                             'SSXBT,SIPPICAN',
                                                                             'T10,SIPPICAN', 
                                                                             'T10,TSK - TSURUMI SEIKI Co.',
                                                                             'T10,UNKNOWN BRAND', 
                                                                             'T11,SIPPICAN',
                                                                             'T11,UNKNOWN BRAND', 
                                                                             'T4,SIPPICAN',
                                                                             'T4,TSK - TSURUMI SEIKI Co.', 
                                                                             'T4,UNKNOWN BRAND']),
                             'instrument_type':numpy.array(['9040 STD', 
                                                            '9060 STD',
                                                            'AXBT',
                                                            'AXBT', 
                                                            'AXBT 536',
                                                            'DEEP BLUE',
                                                            'DEEP BLUE',
                                                            'DEEP BLUE', 
                                                            'FAST DEEP',
                                                            'FAST DEEP',
                                                            'SSXBT',
                                                            'T10', 
                                                            'T10',
                                                            'T10', 
                                                            'T11',
                                                            'T11', 
                                                            'T4',
                                                            'T4', 
                                                            'T4']),
                           'max_depth':numpy.array([numpy.NaN,numpy.NaN,.4,10,1.,2.,3.,4.,numpy.NaN,5,6.,9.,3,numpy.NaN,numpy.NaN,2,3,10,numpy.NaN]),
                           'depth_profile':numpy.array([4,1,2,4,4,4,4,4,0,4,4,2,4,4,4,4,0,0,4]),
                           'platform':numpy.array(['a', numpy.NaN, 'b', 'c', numpy.NaN, 'd', numpy.NaN, 'e', 'f', numpy.NaN, 'g', 'h', numpy.NaN, 'i', 'j', 'k', 'l', 'm', 'n'])}


        self.test_frame = pandas.DataFrame.from_dict(test_dictionary)   
        self.result_frame = pandas.DataFrame.from_dict(result_dictionary)
    
        descriptor_dictionary={'input_features':['max_depth', 'lat', 'lon'],
                               'output_features':['instrument_type', 'instrument_type_and_manifacturer'],
                               'features_to_get_labeled':[],
                               'features_to_get_dummy':[], 
                               'features_to_rescale':[],
                               'features_engineering':{'modulename':'preprocessing.features_engineering',
                                                       'operations':{'1':{'function_name':'probe_type_output',
                                                                         'parameters':{'column':'instrument'}},
                                                                    '2':{'function_name':'map_attributes_to_NaN',
                                                                         'parameters':{'column':'max_depth',
                                                                                       'operator':'<',
                                                                                       'target_value':'0'}},
                                                                    '3':{'function_name':'map_attributes_to_NaN',
                                                                         'parameters':{'column':'platform',
                                                                                       'operator':'==',
                                                                                       'target_value':'O'}},
                                                                    '4':{'function_name':'convert_list_to_integer',
                                                                         'parameters':{'column':'depth_profile'}}}}}
    def test_probe_type_output(self):
        """Test correct creation of output target features"""
        
        probe_type_output(self.test_frame,'instrument')
        
        for key in ['instrument_type','instrument_type_and_manifacturer']:
            pandas.testing.assert_series_equal(self.test_frame[key],self.result_frame[key])
            
    def test_map_attributes_to_NaN(self):
        """Test correct mapping to NaN"""
        
        key = 'max_depth'
        map_attribute_value_to_NaN(self.test_frame, key, '<', '0')
        pandas.testing.assert_series_equal(self.test_frame[key],self.result_frame[key])
        
        key = 'platform'
        map_attribute_value_to_NaN(self.test_frame, key, '==', '\'O\'')
        pandas.testing.assert_series_equal(self.test_frame[key],self.result_frame[key])
        
    def test_convert_list_to_integer(self):
        """Test correct mapping of list to integer number"""
        
        key = 'depth_profile'
        convert_list_to_integer(self.test_frame, key)
        pandas.testing.assert_series_equal(self.test_frame[key],self.result_frame[key])
