import argparse
import ast
import copy
import importlib
import json
import numpy
import os
import pandas
import time
import sklearn.metrics

from preprocessing.data_preprocessing import  DataPreprocessor
pandas.options.mode.chained_assignment = None

import dataexploration.xbt_dataset
from classification.imeta import imeta_classification, XBT_MAX_DEPTH

class NumpyEncoder(json.JSONEncoder):
    """Encoder to print the result of tuning into a json file"""
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif type(obj) in [numpy.int64, numpy.int32, numpy.int16, numpy.int8]:
            return int(obj)
        elif type(obj) in [numpy.float64, numpy.float32]:
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class ClassificationExperiment(object):
    """
    Class designed for implementing features engineering, design of the input space, algorithms fine tuning and delivering outut prediction
    """
    def __init__(self, json_descriptor, data_dir, output_dir, results_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.json_descriptor = json_descriptor
        self.primary_keys = ['learner', 'input_features', 'output_feature','year_range', 'tuning', 'split']
        self.dataset = None
        self.read_json_file()
        
    def run_single_experiment(self):

        print('loading dataset')
        self.load_dataset()
        # get train/test/unseen sets
        print('generating splits')
        X_dict, y_dict, df_dict = self.get_train_test_unseen_sets()
            
        # fit classifier
        print('training classifier')
        clf1 = self.train_classifier(X_dict['train'], y_dict['train'])
        
        # generate scores
        print('generating metrics')
        metrics_train = self.generate_metrics(clf1, df_dict['train'], 'train')        
        metrics_test = self.generate_metrics(clf1, df_dict['test'], 'test')
        exp_results = pandas.merge(metrics_train, metrics_test, on='year')
        metrics_unseen = self.generate_metrics(clf1, df_dict['unseen'], 'unseen')
        exp_results = exp_results.merge(metrics_unseen, on='year')
        
        # generate imeta
        exp_results = exp_results.merge(self.score_imeta(df_dict['train'], 'train'), on='year')
        exp_results = exp_results.merge(self.score_imeta(df_dict['test'], 'est'), on='year')
        exp_results = exp_results.merge(self.score_imeta(df_dict['unseen'], 'unseen'), on='year')
        
        # TODO: run prediction on full dataset
        # TODO: output predictions
        return exp_results
    
                
    def load_dataset(self):
        """
        create a XBTDataset
        only load the specified input and target features, taken from the parameters JSON file
        """
        self.dataset = dataexploration.xbt_dataset.XbtDataset(self.data_dir, self.year_range)
        self.xbt_labelled = self.dataset.filter_obs({'labelled': 'labelled'})
        
        # initialise the feature encoders on the labelled data
        _ = self.xbt_labelled.get_ml_dataset(return_data=False)

        
    def read_json_file(self):
        """Open json descriptor file, load its content into a dictionary"""
        
        if not os.path.isfile(self.json_descriptor):
            raise ValueError('Missing json descriptor!')
        with open(self.json_descriptor) as json_data:
            dictionary = json.load(json_data)
            
        # Check that the input dictionary contains primary keys defined when the class is instantiated
        for k1 in self.primary_keys:
            if k1 not in dictionary.keys():
                raise ValueError(f'Primary key {k1} in not found in json parameter descriptor!')
        
        self.json_params = dictionary
        self.year_range = (self.json_params['year_range'][0], self.json_params['year_range'][1])
        self.input_features = self.json_params['input_features']
        self.target_feature = self.json_params['output_feature']
        self.test_fraction = self.json_params['split']['test_fraction']
        self.train_fraction = 1.0 - self.test_fraction
        self.unseen_fraction = self.json_params['split']['unseen_fraction']
        self.unseen_feature = self.json_params['split']['unseen_feature']
        self.balance_features = self.json_params['split']['balance_features']
        
        learner_dictionary = self.json_params['learner']
        learner_module_name = learner_dictionary['module_name']
        learner_class_name =  learner_dictionary['python_class']
        learner_module = importlib.import_module(learner_module_name)
        learner_class = getattr(learner_module, learner_class_name)
        self.classifier_class = learner_class       
        self.classifier_opts = {}
        self._tuning_dict = self.json_params['tuning']


    def optimize_and_predict(self, out_path='.', year=''):
        """Get information about the learner to be used, the hyperparameters to be tuned, and the use grid-search"""
        

        
        grid_search_parameters = self.dictionary['tuning']
        grid_search_parameters['estimator'] = learner

        for key, item in grid_search_parameters['param_grid'].items():
            if isinstance(item, str):
                grid_search_parameters['param_grid'][key] = ast.literal_eval(item)
        print('Initializing grid search')
        grid_search = GridSearchCV(**grid_search_parameters)
       
        classification_result = {}
        
        sub_directory_name = os.path.basename(self.json_descriptor).split('.')[0]
        
        print('Starting tuning procedure')
        for y_train, y_test, output_target in zip(self.y_trains, self.y_tests, self.dictionary['output_features']):
          
            start = time.time()
            grid_search.fit(self.X_train, y_train)
            end = time.time()

            print('Tuning procedure for '+ output_target+' completed, time elapsed = '+str(end-start)+' seconds.')
            prediction_probabilities = grid_search.predict_proba(self.X_test)
            prediction = grid_search.predict(self.X_test)
            classification_accuracy_score = accuracy_score(prediction, y_test)
            classification_recall_score = recall_score(prediction, y_test, average='weighted')            
            
            classification_result['probabilities'] = prediction_probabilities.mean(axis=0)
            classification_result['class_mapping'] = self.output_maps[output_target]
            classification_result['accuracy'] = classification_accuracy_score
            classification_result['recall'] = classification_recall_score
            classification_result['rescale_all'] = self.dictionary['rescale_all']
            classification_result['input_features'] = self.dictionary['input_features']
            classification_result['applied_operations'] = list(self.dictionary['operations'].keys())
            classification_result['best_hyperparameters'] = grid_search.best_params_
             
            self.generate_results(output_target, out_path, learner_class_name, sub_directory_name, grid_search.cv_results_, classification_result, year)
             
                        
    def generate_results(self, output_target, out_path, learner_class_name, sub_directory_name, tuning_result, classification_result, year):
        """Store results from tuning and prediction into json files, generating a proper tree of directories"""
        
        base_dir = os.path.join(out_path, learner_class_name)
        out_dir = os.path.join(base_dir,sub_directory_name)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        tuning_file_name = os.path.join(out_dir,year+'_'+output_target+'_tuning.json')
        prediction_file_name = os.path.join(out_dir,year+'_'+output_target+'_prediction.json')
        
        for name, data in zip([tuning_file_name, prediction_file_name], [tuning_result, classification_result]):
            with open(name, 'w') as fp:
                json.dump(data, fp, cls=NumpyEncoder)
                

    
    
    def get_train_test_unseen_sets(self):
        unseen_cruise_numbers = self.xbt_labelled.sample_feature_values(self.unseen_feature, fraction=self.unseen_fraction)    
        if self.unseen_feature:
            xbt_unseen = self.xbt_labelled.filter_obs({self.unseen_feature: unseen_cruise_numbers}, mode='include', check_type='in_filter_set')
            xbt_working = self.xbt_labelled.filter_obs({self.unseen_feature: unseen_cruise_numbers}, mode='exclude', check_type='in_filter_set')
        else:
            xbt_unseen = None
            xbt_working = self.xbt_labelled
        xbt_train_all, xbt_test_all = xbt_working.train_test_split(refresh=True, 
                                                                   features=self.balance_features,
                                                                  train_fraction=self.train_fraction)
        X_train_all = xbt_train_all.filter_features(self.input_features).get_ml_dataset()[0]
        X_test_all = xbt_test_all.filter_features(self.input_features).get_ml_dataset()[0]
        y_train_all = xbt_train_all.filter_features([self.target_feature]).get_ml_dataset()[0]
        y_test_all = xbt_test_all.filter_features([self.target_feature]).get_ml_dataset()[0]
        if xbt_unseen:
            X_unseen_all = xbt_unseen.filter_features(self.input_features).get_ml_dataset()[0]
            y_unseen_all = xbt_unseen.filter_features([self.target_feature]).get_ml_dataset()[0]    
        else:
            X_unseen_all = None
            y_unseen_all = None
        df_dict = {'train': xbt_train_all,
                  'test': xbt_test_all,
                  'unseen': xbt_unseen,}
        X_dict = {'train': X_train_all,
                  'test': X_test_all,
                  'unseen': X_unseen_all,
                 }
        y_dict = {'train': y_train_all,
                  'test': y_test_all,
                  'unseen': y_unseen_all,
                 }
        return X_dict, y_dict, df_dict                

    def train_classifier(self, X_train, y_train):
        clf = self.classifier_class(**self.classifier_opts)
        clf.fit(X_train,y_train)
        return clf    
    
    def score_year(self, xbt_df, year, clf):
        X_year = xbt_df.filter_obs({'year': year}, ).filter_features(self.input_features).get_ml_dataset()[0]
        y_year = xbt_df.filter_obs({'year': year} ).filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_year = clf.predict(X_year)
        metric_year = sklearn.metrics.precision_recall_fscore_support(
            y_year, y_res_year, average='micro')
        return metric_year    

    def generate_imeta(self, xbt_ds):
        imeta_classes = xbt_ds.xbt_df.apply(imeta_classification, axis=1)
        imeta_df = pandas.DataFrame.from_dict({
            'instrument': imeta_classes.apply(lambda t1: f'XBT: {t1[0]} ({t1[1]})'),
            'model': imeta_classes.apply(lambda t1: t1[0]),
            'manufacturer': imeta_classes.apply(lambda t1: t1[1]),
        })
        return imeta_df
    
    def score_imeta(self, xbt_ds, data_label):
        imeta_scores = []
        for year in range(self.year_range[0],self.year_range[1]):
            xbt_year = xbt_ds.filter_obs({'year': year} )
            y_year = xbt_year.filter_features([self.target_feature]).get_ml_dataset()[0]
            imeta_df_year = self.generate_imeta(xbt_year)
            y_imeta = xbt_ds._feature_encoders[self.target_feature].transform(imeta_df_year[[self.target_feature]])

            cats = list(xbt_ds._feature_encoders[self.target_feature].categories_[0])
            prec_year, recall_year, f1_year, support_year = sklearn.metrics.precision_recall_fscore_support(
                y_year, y_imeta, average='micro')
            prec_cat, recall_cat, f1_cat, support_cat = sklearn.metrics.precision_recall_fscore_support(
                y_year, y_imeta)

            metric_dict = {
                'year': year,
                'precision_imeta_{dl}_all'.format(dl=data_label): prec_year,
                'recall_imeta_{dl}_all'.format(dl=data_label): recall_year,
                'f1_imeta_{dl}_all'.format(dl=data_label): f1_year,
                          }
            metric_dict.update({'precision_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, prec_cat)})
            metric_dict.update({'recall_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, recall_cat)})
            metric_dict.update({'f1_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, f1_cat)})
            metric_dict.update({'support_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, support_cat)})
            imeta_scores += [metric_dict]
        
        return pandas.DataFrame.from_records(imeta_scores)
        
    def generate_metrics(self, clf, xbt_ds, data_label):
        metric_list = []
        for year in range(self.year_range[0],self.year_range[1]):
            X_year = xbt_ds.filter_obs({'year': year}, ).filter_features(self.input_features).get_ml_dataset()[0]
            y_year = xbt_ds.filter_obs({'year': year} ).filter_features([self.target_feature]).get_ml_dataset()[0]

            y_res_year = clf.predict(X_year)
            cats = list(xbt_ds._feature_encoders[self.target_feature].categories_)[0]
            prec_year, recall_year, f1_year, support_year = sklearn.metrics.precision_recall_fscore_support(
                y_year, y_res_year, average='micro')
            prec_cat, recall_cat, f1_cat, support_cat = sklearn.metrics.precision_recall_fscore_support(
                y_year, y_res_year)

            column_template = '{metric}_{data}_{subset}'
            metric_dict = {'year': year,
                           column_template.format(data=data_label, metric='precision', subset='all'): prec_year,
                           column_template.format(data=data_label, metric='recall', subset='all'): recall_year,
                           column_template.format(data=data_label, metric='f1', subset='all'): f1_year,
                          }

            metric_dict.update({column_template.format(data=data_label, metric='precision', subset=cat): val for cat, val in zip(cats, prec_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='recall', subset=cat): val for cat, val in zip(cats, recall_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='f1', subset=cat): val for cat, val in zip(cats, f1_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='support', subset=cat): val for cat, val in zip(cats, support_cat)})
            metric_list += [metric_dict]
            
        metrics_df = pandas.DataFrame.from_records(metric_list)
        #TODO: write to file
        return metrics_df        
    
    def generate_prediction(self):
       # add a column to the dataframe representing the prediction
        pass
        
    def generate_vote_probabilities(self):
        # take a list of estimators
        # generate a prediction for each
        # sum the predictions from classifiers for each class for each obs
        # generate a one hot style probability of each class based by normalising the vote counts to sum to 1 (divide by num estimators)
        pass
        
    def generate_vote_probabilities(self):
        # generate probailities directly from the estimator
        pass
        
    def output_predictions(self):
        # write_predictions to a CSV file
        # user specifies which features to write
        pass
        


    
def run_experiment(data_path, result_path, year, train_set_name, test_set_name, json_descriptor):
    """Run a single classification experiment"""
    
    train_file_name = os.path.join(data_path, train_set_name)
    test_file_name = os.path.join(data_path, test_set_name)
    
    experiment = ClassificationExperiment(train_file_name, test_file_name, json_descriptor)
    experiment.read_json_file()
    experiment.get_datasets()
    experiment.remove_useless_features()
    experiment.apply_operations()   
    experiment.get_class_codes_maps()
    experiment.imputation()
    experiment.categorical_features_conversion()
    experiment.rescale()
    experiment.generate_train_test_arrays()
    experiment.optimize_and_predict(result_path, year)
    
def run_single_experiment(data_dir, output_dir, year_range):
    experiment = ClassificationExperiment(train_file_name, test_file_name, json_descriptor)
    pass
    
def run_probability_experiment(data_dir, output_dir, year_range):
    pass

def run_cvhpt_experiment(data_dir, output_dir, year_range):
    # create the experiment
    # add fucntions to expriment class to:
    # load the dataset
    # load experiment parameters from json
    # get labelled
    # create inner and outer fold parameters
    # create encoders
    # create hp tune cv tune object
    # setup parameters for outer cross validation
    # run cross validation and hp tuning
    # run scoring for each of the estimators from outer CV
    # generate prediction for each estimator
    # output all predictions
    # using voting to generate voting based probabilities
    pass    
    
    
def main():
    """For running a specific classification experiment from shell"""
    
    parser = argparse.ArgumentParser(description='Run a specific classification experiment')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='outputpath for results')
    parser.add_argument('--year', default='',help='year')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('json_descriptor',help='json descriptor file name')
    
    args = parser.parse_args()

    run_experiment(args.path, args.outpath, args.year, args.train, args.test, args.json_descriptor)

    
if __name__ == "__main__":
    # execute only if run as a script
    main()