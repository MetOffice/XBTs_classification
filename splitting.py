""" Script for redefining the train-test split, and for subsampling them according the value of the year label """

import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing

def main():

    parser = argparse.ArgumentParser(description='Redefine train-test split.\n Remove irrelevant input features. Produce a train/set dataset for each year')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='output train and test files location')
    parser.add_argument('-startyear',default=str(1970),help='start year')    
    parser.add_argument('-endyear',default=str(2017),help='end year')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
        
    args = parser.parse_args()

    train_file_name = args.path+args.train
    test_file_name = args.path+args.test
    
    train = pd.read_csv(train_file_name)
    test = pd.read_csv(test_file_name)

    # concatenate train and test set, removing useless features
    data = pd.concat([train,test])
    useless_features = ['Unnamed: 0','cruise_number','id','depth_profile','temperature_profile']
    data=data.drop(useless_features,axis=1) 

    # transforming date into string type
    data['date']=data['date'].astype(str)
    data['date']=data['date'].str.slice(0,4)

    # transforming categorical data by applying an integer mapping
    encoder = preprocessing.LabelEncoder()
    categorical_features = ['country','institute','instrument','platform']
    for key in categorical_features:
        encoder.fit(data[key])
        transformed=encoder.transform(data[key])
        data[key].iloc[:]=transformed
    
    # reshuffling
    data = data.reset_index()
    data = data.reindex(np.random.permutation(data.index))
    size=data.shape[0]

    # splitting
    percentage = .75
    size_new_train = int(size*percentage)
    new_train = data.iloc[0:size_new_train,:]
    new_test = data.iloc[size_new_train:,:]

    # saving output
    for year in range(int(args.startyear),int(args.endyear)+1):
        train_year = new_train[new_train['date']==str(year)]
        test_year = new_test[new_test['date']==str(year)]
        for name,data_frame in zip(['train_'+str(year)+'.csv','test_'+str(year)+'.csv'],[train_year,test_year]):
            out_file_name = args.outpath+name
            data_frame.to_csv(out_file_name)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
