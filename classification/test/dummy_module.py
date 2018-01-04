"""Dummy module for testing the correctenss of the \'apply_operations\' contained in the classification module"""

def change_value(dataframe, column, operator, target_value, new_value):
    """Convert a specific input feature value into a new value."""
    
    operation = 'lambda s: s'+operator+target_value
    research_index = dataframe[column].loc[eval(operation)].index
    dataframe[column].iloc[research_index] = new_value
    
def new_feature_creation(dataframe, left_column, right_column, filling_value):
    """Creates a new feature column by using the combined labels of left and right columns, then fill it with filling_value"""

    dataframe[left_column+'_'+right_column]=filling_value

