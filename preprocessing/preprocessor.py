"""
Redefine train-test split. Apply some data preprocessing if required from user. Produce a train/set dataset for each year
"""

import argparse
from preprocessing.data_preprocessing import  DataPreprocessor

def main():

    parser = argparse.ArgumentParser(description='Redefine train-test split.\n Apply features engineering. Produce a train/set dataset for each year')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='output train and test files location')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('-startyear',default=str(1966),help='start year')    
    parser.add_argument('-endyear',default=str(2017),help='end year')
    parser.add_argument('-useless_features',nargs='+',default=[],help='features to be removed')    
    parser.add_argument('-features_to_get_labeled',nargs='+',default=[],help='categorical features to be converted into ordinal features')
    parser.add_argument('-features_to_get_dummy',nargs='+',default=[],help='categorical features to be converted into dummy variables')
    parser.add_argument('-reshuffle',default=True,help='boolean flag to activate reshuffling')    
    parser.add_argument('-train_split',default=.75,type=float,help='proportion of original data to be retained for training')
    parser.add_argument('-test_split',default=.25,type=float,help='proportion of original data to be retained for testing')
    args = parser.parse_args()

    train_file_name = args.path+args.train
    test_file_name = args.path+args.test
    
    if args.reshuffle== 'True':
        reshuffle = True
    elif args.reshuffle == 'False':
        reshuffle = False
    
    # initializing data preprocessor 
    data_preprocessor =  DataPreprocessor(train_file_name, 
					  test_file_name, 
					  useless_features=args.useless_features, 
					  reshuffle=reshuffle, 
					  train_split=args.train_split, 
					  test_split=args.test_split)
    
    data = data_preprocessor.load_data(stack=True)
    data_preprocessor.remove_useless_features(data)
    data_preprocessor.recode_column_values(data,'date',0,4)
    new_train, new_test = data_preprocessor.split_to_train_test(data)

    data_preprocessor.print_new_datasets_to_csv(new_train, new_test, args.outpath,int(args.startyear),int(args.endyear))

if __name__ == "__main__":
    # execute only if run as a script
    main()