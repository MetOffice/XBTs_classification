import numpy 
import pandas 
from sklearn import preprocessing

class DataPreprocessor(object):

    @staticmethod
    def sanity_check(element,element_type):
	"""check that the given element is the correct instance of element_type"""

	if not isinstance(element,element_type):
	    raise ValueError("element must be a ",element_type," object.")

    def __init__(self, train_dataset_name, test_dataset_name, useless_features, features_to_get_labeled, features_to_get_dummy, reshuffle=True, train_split=.75, test_split=.25):
	self.train_dataset_name = train_dataset_name
	self.test_dataset_name = test_dataset_name
	self.useless_features = useless_features
	self.features_to_get_labeled = features_to_get_labeled
	self.features_to_get_dummy = features_to_get_dummy
	self.reshuffle = reshuffle
	self.train_split = train_split
	self.test_split = test_split

	# basic initialization checks
	list_of_checks = [str,str,list,list,list,bool,float,float]
	list_of_elements = [self.train_dataset_name, self.test_dataset_name, self.useless_features, self.features_to_get_labeled, self.features_to_get_dummy, self.reshuffle, self.train_split, self.test_split]

	for element, element_check_type in zip(list_of_elements,list_of_checks):
	    self.sanity_check(element,element_check_type)

    def load_data(self):
	"""load train and test dataset, concatenate them into a large dataset, convert date from float to string"""

	train_dataset = pandas.read_csv(self.train_dataset_name)
	test_dataset = pandas.read_csv(self.test_dataset_name)
	self.data = pandas.concat([train_dataset,test_dataset])

	# transforming date into string type
	self.data['date']=self.data['date'].astype(str)
	self.data['date']=self.data['date'].str.slice(0,4)

    def remove_useless_features(self):
	"""removes features not used for the classification procedure"""

	self.data=self.data.drop(self.useless_features,axis=1)

    def categorical_features_to_label(self):
	"""convert categorical features to ordinal integers, by applying a univoque mapping"""

	encoder = preprocessing.LabelEncoder()
	for key in self.features_to_get_labeled:
	    encoder.fit(self.data[key])
	    transformed=encoder.transform(self.data[key])
	    self.data[key].iloc[:]=transformed

    def categorical_features_to_dummy(self):
	"""convert categorical features to dummy variables"""
	
	pandas.get_dummies(self.data)
    
    def split_to_train_test(self): 
	"""split large dataset, into new train and test set, with train/test size determined by train_split/test_split attributes. Apply reshuffling if reshuffle=True"""

	if self.reshuffle:
	  self.data = self.data.reset_index()
	  self.data = self.data.reindex(numpy.random.permutation(self.data.index))
      
	size=self.data.shape[0]
	size_new_train = int(size*self.train_split)
	self.new_train = self.data.iloc[0:size_new_train,:]
	self.new_test = self.data.iloc[size_new_train:,:]

    def print_new_datasets_to_csv(self,outpath,startyear,endyear):
	"""extract a single dataset from new train and test dataset, by matching a specific year value. Saves the datasets to csv files. Repeat the operation by looping over a range of years"""

	for year in range(startyear,endyear+1):
	    train_year = self.new_train[self.new_train['date']==str(year)]
	    test_year = self.new_test[self.new_test['date']==str(year)]
	    for name,data_frame in zip(['train_'+str(year)+'.csv','test_'+str(year)+'.csv'],[train_year,test_year]):
		out_file_name = outpath+name
		data_frame.to_csv(out_file_name)