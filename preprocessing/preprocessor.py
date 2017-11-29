"""Redefine train-test split. Apply features engineering. Produce a train/set dataset for each year"""

import argparse
from data_preprocessing import  DataPreprocessor

def main():

    parser = argparse.ArgumentParser(description='Redefine train-test split.\n Apply features engineering. Produce a train/set dataset for each year')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='output train and test files location')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('-startyear',default=str(1970),help='start year')    
    parser.add_argument('-endyear',default=str(2017),help='end year')
    parser.add_argument('-useless_features',nargs='+',default=[],help='features to be removed')    
    parser.add_argument('-features_to_get_labeled',nargs='+',default=[],help='categorical features to be converted into ordinal features')
    parser.add_argument('-features_to_get_dummy',nargs='+',default=[],help='categorical features to be converted into dummy variables')
    parser.add_argument('-reshuffle',default=True,help='boolean flag to activate reshuffling')    
    parser.add_argument('-train_split',default=.75,help='proportion of original data to be retained for training')
    parser.add_argument('-test_split',default=.25,help='proportion of original data to be retained for testing')
    args = parser.parse_args()

    train_file_name = args.path+args.train
    test_file_name = args.path+args.test

    # initializing data preprocessor 
    data_preprocessor =  DataPreprocessor(train_file_name, 
					  test_file_name, 
					  args.useless_features, 
					  args.features_to_get_labeled, 
					  args.features_to_get_dummy, 
					  args.reshuffle, 
					  args.train_split, 
					  args.test_split)
    
    data.load_data()
    
    data.remove_useless_features()
    
    data.categorical_features_to_label()

    data.categorical_features_to_dummy()

    data.split_to_train_test()

    data.print_new_datasets_to_csv(ars.outpath,int(args.startyear),int(args.endyear))
