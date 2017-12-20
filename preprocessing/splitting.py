""" Script for redefining the train-test split, and for subsampling them according the value of the year label """

import argparse
from data_preprocessing import DataPreprocessor

def main():

    parser = argparse.ArgumentParser(description='Redefine train-test split.\n Remove irrelevant input features. Produce a train/set dataset for each year')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='output train and test files location')
    parser.add_argument('-startyear',default=str(1970),help='start year')    
    parser.add_argument('-endyear',default=str(2017),help='end year')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('useless_features',nargs='+', help='features not to be used')
    parser.add_argument('reshuffle',default=False, help='split flag')
    parser.add_argument('train_split',default=.75, help='proportion of data to be retained for training(as a decimal between 0 and 1)')
    parser.add_argument('test_split',default=.25, help='proportion of data to be retained for testing(as a decimal between 0 and 1)')        
        
    args = parser.parse_args()

    train_file_name = args.path+args.train
    test_file_name = args.path+args.test
    
    preprocessor = DataPreprocessor(train_file_name, test_file_name, useless_features=args.useless_features, reshuffle=args.reshuffle, train_split=args.train_split, test_split=args.test_split)

    # concatenate train and test set, removing useless features
    data = preprocessor.load_data(stack=True)
    preprocessor.remove_useless_features(data)

    # transforming date into string type
    preprocessor.recode_column_values(data,'date',0,4)
    
    new_train_dataset, new_test_dataset = preprocessor.split_to_train_test(data)
    preprocessor.print_new_datasets_to_csv(new_train_dataset, new_test_dataset, args.outpath, args.startyear, args.endyear)    

if __name__ == "__main__":
    # execute only if run as a script
    main()
