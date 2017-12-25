"""Collection of methods implementing features engineering for the XBTs classification project"""

import json
import numpy

def apply_features_engineering(dataframe,json_descriptor):
    operations = load_operations(json_descriptor)
    return operations    
    
def load_operations(filename):
    # Reading data
    with open(filename, 'r') as f:
     data = json.load(f)
     print(data)


def probe_type_output(dataframe, column):
    """Redefine the target outputs as \"instrument_type\" and \"instrument_type_and_manifacturer\"."""

    if column != 'instrument':
        raise ValueError
    else:
        # eliminate suffix "XBT/STD: "
        dataframe[column]=dataframe[column].str.slice(5,) 
        
        # get rid of brackets
        dataframe[column]=dataframe[column].str.replace(')','')
        dataframe[column]=dataframe[column].str.replace('(',',')

        # some minor operations
        dataframe[column]=dataframe[column].str.replace(' UNKNOWN BRAND','UNKNOWN BRAND')
        dataframe[column]=dataframe[column].str.replace('DEEP BLUE ','DEEP BLUE')
        dataframe[column]=dataframe[column].str.replace('FAST DEEP ','FAST DEEP')
        dataframe[column]=dataframe[column].str.replace(' ,',',')        
        dataframe[column]=dataframe[column].str.replace(', ',',')      
        dataframe[column]=dataframe[column].str.replace('SUBMARINE-LAUNCHED EXPENDABLE BATHYTHERMOGRAPH,','')
        
        # instrument type
        dataframe[column+'_type']=dataframe[column].str.split(',').str.get(-2)
        
        # instrument and manifacturer
        dataframe.rename(columns={column: column+'_type_and_manifacturer'}, inplace=True)
        
def map_attribute_value_to_NaN(dataframe, column, operator, target_value):
    """Convert a specific input feature value into numpy.Nan."""
    
    operation = 'lambda s: s'+operator+target_value
    research_index = dataframe[column].loc[eval(operation)].index
    numeric = dataframe._get_numeric_data().columns
    if column in numeric:
        dataframe[column].iloc[research_index] = numpy.nan
    else:
        dataframe[column].iloc[research_index] = 'nan'
        
def convert_list_to_integer(dataframe, column):
    """Convert an input feature from list to integer"""
    if column != 'depth_profile':
        raise ValueError
        
    dataframe[column]=dataframe[column].str.count('\.')
    