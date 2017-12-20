import numpy 
import pandas 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DataPreprocessor(object):

    @staticmethod
    def sanity_check(element,element_type):
        """check that the given element is the correct instance of element_type"""

        if not isinstance(element,element_type):
            raise ValueError("element must be a ",element_type," object.")

    def __init__(self, train_dataset_name, test_dataset_name, useless_features=[], features_to_get_labeled=[], features_to_get_dummy=[],features_to_rescale=[], reshuffle=True, train_split=.75, test_split=.25):
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.useless_features = useless_features
        self.features_to_get_labeled = features_to_get_labeled
        self.features_to_get_dummy = features_to_get_dummy
        self.features_to_rescale = features_to_rescale
        self.reshuffle = reshuffle
        self.train_split = train_split
        self.test_split = test_split

        # basic initialization checks
        list_of_checks = [str,str,list,list,list,list,bool,float,float]
        list_of_elements = [self.train_dataset_name, self.test_dataset_name, self.useless_features, self.features_to_get_labeled, self.features_to_get_dummy, self.features_to_rescale, self.reshuffle, self.train_split, self.test_split]

        for element, element_check_type in zip(list_of_elements,list_of_checks):
            self.sanity_check(element,element_check_type)

    def load_data(self,stack = False):
        """load and then return train and test dataset: if flag stack enabled, then concatenate them into a large dataset and return it"""
            
        train_dataset = pandas.read_csv(self.train_dataset_name)
        test_dataset = pandas.read_csv(self.test_dataset_name)
        
        if stack:        
            return pandas.concat([train_dataset,test_dataset])
        else:
            return train_dataset, test_dataset

    @staticmethod        
    def recode_column_values(dataframe,column,start,end):
        """convert column values into string type, keep only characters within the interval defined by start and end """
        
        dataframe[column]=dataframe[column].astype(str)
        dataframe[column]=dataframe[column].str.slice(start,end)

    def remove_useless_features(self,dataframe):
        """removes features not used for the classification procedure"""

        dataframe.drop(labels=self.useless_features,axis=1,inplace=True)

    def categorical_features_to_label(self,dataframe):
        """convert categorical features to ordinal integers, by applying a univoque mapping"""

        encoder = LabelEncoder()
        for key in self.features_to_get_labeled:
            encoder.fit(dataframe[key])
            transformed=encoder.transform(dataframe[key])
            dataframe[key].iloc[:]=transformed

    def categorical_features_to_dummy(self,dataframe):
        """convert categorical features to dummy variables"""

        converted = pandas.get_dummies(dataframe[self.features_to_get_dummy])
        dataframe.drop(self.features_to_get_dummy,axis=1,inplace = True)
        for key in converted.columns:
            dataframe[key] =  converted[key]
            
    def impute_numerical_nans(self,dataframe_1, dataframe_2, strategy, axis, missing_values = 'Nan', as_frame = True):
        """Fill Nans in numerical input features"""
        
        imputer = Imputer(missing_values=missing_values, strategy=strategy, axis=axis, verbose=0, copy=False)
        X_1, X_2 = dataframe_1.values, dataframe_2.values
        imputer.fit_transform(X_1)
        imputer.transform(X_2)
        
        if as_frame:
            new_train_frame = pandas.DataFrame(X_1,columns = dataframe_1.columns)
            new_test_frame = pandas.DataFrame(X_2,columns = dataframe_2.columns)
            return new_train_frame,new_test_frame
        else:
            return X_1,X_2

            
    def rescale_features(self, dataframe_1, dataframe_2, as_frame = True):
        """Rescale input features by removing the mean and normalizing with respect the variance"""
    
        rescaler = StandardScaler()
        X_1, X_2 = dataframe_1.values, dataframe_2.values
        rescaler.fit_transform(X_1)
        rescaler.transform(X_2)
        
        if as_frame:
            return dataframe_1, dataframe_2
        else:
            return X_1,X_2
        
    def split_to_train_test(self,dataframe): 
        """split large dataset, into new train and test set, with train/test size determined by train_split/test_split attributes. Apply reshuffling if reshuffle flag enabled"""

        dataframe.reset_index(inplace=True,drop=True)
        
        if self.reshuffle:
            dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
      
        size=dataframe.shape[0]
        size_new_train = int(size*self.train_split)
        new_train_dataset = dataframe.iloc[0:size_new_train,:]
        new_test_dataset = dataframe.iloc[size_new_train:,:]
        
        return new_train_dataset,new_test_dataset

    def print_new_datasets_to_csv(self,new_train_dataset,new_test_dataset,outpath,startyear,endyear):
        """extract a single dataset from new train and test dataset, by matching a specific year value. Saves the datasets to csv files. Repeat the operation by looping over a range of years"""

        for year in range(startyear,endyear+1):
            train_year = new_train_dataset[self.new_train['date']==str(year)]
            test_year = new_test_dataset[self.new_test['date']==str(year)]
            for name,data_frame in zip(['train_'+str(year)+'.csv','test_'+str(year)+'.csv'],[train_year,test_year]):
                out_file_name = outpath+name
                data_frame.to_csv(out_file_name,index=False)
# Still to do: numerical filling of nan, numerical filling of categorical, creation of new features