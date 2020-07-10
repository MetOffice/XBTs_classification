import argparse
import ast
import copy
import datetime
import importlib
import joblib
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


RESULT_FNAME_TEMPLATE = 'xbt_metrics_{name}.csv'
OUTPUT_FNAME_TEMPLATE = 'xbt_classifications_{name}.csv'
CLASSIFIER_EXPORT_FNAME_TEMPLATE = 'xbt_classifier_{exp}_{split_num}.joblib'
DATESTAMP_TEMPLATE = '{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}'

UNSEEN_FOLD_NAME = 'unseen_fold'
RESULT_FEATURE_TEMPLATE = '{target}_res_{clf}_split{split_num}'
PROB_CAT_TEMPLATE = '{target}_{clf}_probability_{cat}'

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
    def __init__(self, json_descriptor, data_dir, output_dir):
        # assign arguments
        self.data_dir = data_dir
        self.root_output_dir = output_dir
        self.json_descriptor = json_descriptor
        
        # initialise to None where appropriate
        self.dataset = None
        self.results = None
        self.classifiers = None
        self._n_jobs = -1
        self._cv_output = None
        self.xbt_predictable = None
        self._exp_datestamp = None
        self.classifier_fnames = None
        self.experiment_description_dir = None
        
        # load experiment definition from json file
        self.primary_keys = ['learner', 'input_features', 'output_feature','year_range', 'tuning', 'split', 'experiment_name']
        self.read_json_file()
        self.exp_output_dir = os.path.join(self.root_output_dir, self.experiment_name)
        
        
    def _generate_exp_datestamp(self):
        self._exp_datestamp = DATESTAMP_TEMPLATE.format(dt=datetime.datetime.now())
        
    def run_single_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        self._check_output_dir()
        self._generate_exp_datestamp()

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
        print('generating imeta output')
        exp_results = exp_results.merge(self.score_imeta(df_dict['train'], 'train'), on='year')
        exp_results = exp_results.merge(self.score_imeta(df_dict['test'], 'est'), on='year')
        exp_results = exp_results.merge(self.score_imeta(df_dict['unseen'], 'unseen'), on='year')
        
        self.results = exp_results
        self.classifiers = {0: clf1}
        
        # output results to a file
        if write_results:
            out_name = self.experiment_name + '_' + self._exp_datestamp
            self.results.to_csv(
                os.path.join(self.exp_output_dir,
                             RESULT_FNAME_TEMPLATE.format(name=out_name)))
        
        
        # generate imeta algorithm results for the whole dataset
        print('generating predictions for the whole dataset.')
        imeta_df = self.generate_imeta(self.dataset)
        imeta_df = imeta_df.rename(
            columns={'instrument': 'imeta_instrument',
                     'model': 'imeta_model',
                     'manufacturer': 'imeta_manufacturer',
            })        
        self.dataset.xbt_df = self.dataset.xbt_df.merge(imeta_df[['id', 'imeta_{0}'.format(self.target_feature)]])        
        
        # run prediction on full dataset
        feature_name = '{target}_res_{name}'.format(target=self.target_feature,
                                                    name=self.classifier_name,
                                                   )
        self.generate_prediction(self.classifiers[0], feature_name)
        
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            out_path = os.path.join(self.exp_output_dir, OUTPUT_FNAME_TEMPLATE.format(name=out_name))
            print(f'output predictions to {out_path}')
            self.dataset.output_data(
                out_path,
                add_ml_features=[feature_name])
            
        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()
            
        return (self.results, self.classifiers)
    
    def run_cvhpt_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        self._check_output_dir()
        self._generate_exp_datestamp()
        
        print('loading dataset')
        self.load_dataset()
        # get train/test/unseen sets
        print('generating splits')
        self.xbt_labelled.generate_folds_by_feature(self.unseen_feature, self.num_unseen_splits, UNSEEN_FOLD_NAME)        
        
        X_labelled = self.xbt_labelled.filter_features(self.input_features).get_ml_dataset()[0]
        y_labelled = self.xbt_labelled.filter_features([self.target_feature]).get_ml_dataset()[0]
        
        # create objects for cross validation and hyperparameter tuning
        # first set up objects for the inner cross validation, which run for each
        # set of hyperparameters in the grid search.
        print('creating hyperparameter tuning objects')

        group_cv1 = sklearn.model_selection.GroupKFold(n_splits=self.num_unseen_splits)
        classifier_obj = self.classifier_class(**self._default_classifier_opts)
        grid_search_cv = sklearn.model_selection.GridSearchCV(
            classifier_obj,
            param_grid = self._tuning_dict['param_grid'],  
            scoring=self._tuning_dict['scoring'],
            cv=self.num_training_splits,
        )
        
        # now run the outer cross-validation with the grid search HPT as the classifier,
        # so for each split, HPT will be run with CV on each set of hyperparameters
        print('running cross validation')
        scores = sklearn.model_selection.cross_validate(
            grid_search_cv,
            X_labelled, y_labelled, 
            groups=self.xbt_labelled[UNSEEN_FOLD_NAME], 
            cv=group_cv1,
            return_estimator=self.return_estimator,
            return_train_score=self.return_train_score,
            scoring=self._tuning_dict['cv_metrics'],
            n_jobs=self._n_jobs,
        )        

        self._cv_output = scores
        self.classifiers = {split_num: est1 
                            for split_num, est1 in enumerate(scores['estimator'])}
       
        print('calculating metrics')
        metrics_list = {}
        for split_num, estimator in self.classifiers.items():
            xbt_train = self.xbt_labelled.filter_obs({UNSEEN_FOLD_NAME: split_num}, mode='exclude')
            xbt_test = self.xbt_labelled.filter_obs({UNSEEN_FOLD_NAME: split_num}, mode='include')
            train_metrics = self.generate_metrics(estimator, xbt_train, 'train_{0}'.format(split_num))
            test_metrics = self.generate_metrics(estimator, xbt_test, 'test_{0}'.format(split_num))
            metrics_list[split_num] = pandas.merge(train_metrics, test_metrics, on='year')        

        metrics_df_merge = None
        for label1, metrics1  in metrics_list.items():
            if metrics_df_merge is None:
                metrics_df_merge = metrics1
            else:
                metrics_df_merge = pandas.merge(metrics_df_merge, metrics1)            
        self.results = metrics_df_merge

        # output results to a file
        if write_results:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            self.results.to_csv(
                os.path.join(self.exp_output_dir,
                             RESULT_FNAME_TEMPLATE.format(name=out_name)))
        
        
        # generate imeta algorithm results for the whole dataset
        imeta_df = self.generate_imeta(self.dataset)
        imeta_df = imeta_df.rename(
            columns={'instrument': 'imeta_instrument',
                     'model': 'imeta_model',
                     'manufacturer': 'imeta_manufacturer',
            })
        self.dataset.xbt_df = self.dataset.xbt_df.merge(imeta_df[['id', 'imeta_{0}'.format(self.target_feature)]])
        
        # run prediction on full dataset
        result_feature_names = []
        for split_num, estimator in self.classifiers.items():
            res_name = RESULT_FEATURE_TEMPLATE.format(
                target=self.target_feature,
                clf=self.classifier_name,
                split_num=split_num)
            result_feature_names += [res_name]        
            self.generate_prediction(estimator, res_name)
        
        # generate vote count probabilities from the different trained classifiers
        self.generate_vote_probabilities(result_feature_names)
        
        # output predictions
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            out_path = os.path.join(self.exp_output_dir, OUTPUT_FNAME_TEMPLATE.format(name=out_name))
            print(f'output predictions to {out_path}')
            self.dataset.output_data(
                out_path,
                add_ml_features=[])
                    
        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()
            
        return (self.results, self.classifiers)
    
    def run_inference(self, write_predictions=True):
        """
        """
        self._check_output_dir()
        self._generate_exp_datestamp()
        
        print('loading dataset')
        self.load_dataset()

        print('loading saved classifiers from pickle files.')
        self.classifiers = {ix1: joblib.load(os.path.join(self.experiment_description_dir,
                                                         fname1))
                            for ix1, fname1 in enumerate(self.classifier_fnames)
                           }
        
        print('generate imeta algorithm results for the whole dataset')
        imeta_df = self.generate_imeta(self.dataset)
        imeta_df = imeta_df.rename(
            columns={'instrument': 'imeta_instrument',
                     'model': 'imeta_model',
                     'manufacturer': 'imeta_manufacturer',
            })
        self.dataset.xbt_df = self.dataset.xbt_df.merge(imeta_df[['id', 'imeta_{0}'.format(self.target_feature)]])
        
        print(' run prediction on full dataset')
        result_feature_names = []
        for split_num, estimator in self.classifiers.items():
            res_name = RESULT_FEATURE_TEMPLATE.format(
                target=self.target_feature,
                clf=self.classifier_name,
                split_num=split_num)
            result_feature_names += [res_name]        
            self.generate_prediction(estimator, res_name)
        
        print('generate vote count probabilities from the different trained classifiers')
        self.generate_vote_probabilities(result_feature_names)
        
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            out_path = os.path.join(self.exp_output_dir, OUTPUT_FNAME_TEMPLATE.format(name=out_name))
            print(f'output predictions to {out_path}')
            self.dataset.output_data(
                out_path,
                add_ml_features=[])
                    
        return self.classifiers
    
    
    def load_dataset(self):
        """
        create a XBTDataset
        only load the specified input and target features, taken from the parameters JSON file
        """
        self.dataset = dataexploration.xbt_dataset.XbtDataset(self.data_dir, self.year_range)
        self.xbt_labelled = self.dataset.filter_obs({'labelled': 'labelled'})
        
        # initialise the feature encoders on the labelled data
        _ = self.xbt_labelled.get_ml_dataset(return_data=False)

    def _check_output_dir(self):
        if not self.exp_output_dir:
            raise RuntimeError(f'experiment output directory path ({self.exp_output_dir}) not defined.')
        
        if not os.path.isdir(self.exp_output_dir):
            os.makedirs(self.exp_output_dir)
        
    def read_json_file(self):
        """Open json descriptor file, load its content into a dictionary"""
        
        if not os.path.isfile(self.json_descriptor):
            raise ValueError(f'Missing json descriptor {self.json_descriptor}!')
        if not os.path.isabs(self.json_descriptor):
            self.json_descriptor = os.path.abspath(self.json_descriptor)
        self.experiment_description_dir, self.json_fname  = os.path.split(self.json_descriptor)
        with open(self.json_descriptor) as json_data:
            dictionary = json.load(json_data)
            
        # Check that the input dictionary contains primary keys defined when the class is instantiated
        for k1 in self.primary_keys:
            if k1 not in dictionary.keys():
                raise ValueError(f'Primary key {k1} in not found in json parameter descriptor!')
        
        self.json_params = dictionary
        self.experiment_name = self.json_params['experiment_name']
        self.year_range = (self.json_params['year_range'][0], self.json_params['year_range'][1])
        self.input_features = self.json_params['input_features']
        self.target_feature = self.json_params['output_feature']
        
        self.num_training_splits = int(self.json_params['split']['num_training_splits'])
        self.test_fraction = 1.0 / self.num_training_splits
        self.train_fraction = 1.0 - self.test_fraction
        self.num_unseen_splits = int(self.json_params['split']['num_unseen_splits'])
        self.unseen_fraction = 1.0 / self.num_unseen_splits
        self.unseen_feature = self.json_params['split']['unseen_feature']
        self.balance_features = self.json_params['split']['balance_features']
        
        self.return_estimator =  self.json_params['tuning']['return_estimator']
        self.return_train_score =  self.json_params['tuning']['return_train_score']
        
        learner_dictionary = self.json_params['learner']
        learner_module_name = learner_dictionary['module_name']
        learner_class_name =  learner_dictionary['python_class']
        learner_module = importlib.import_module(learner_module_name)
        learner_class = getattr(learner_module, learner_class_name)
        self.classifier_name = learner_dictionary['name']
        self.classifier_class = learner_class       
        self._tuning_dict = self.json_params['tuning']
        try:
            self.classifier_fnames = self.json_params['classifier_fnames']
        except KeyError:
            self.classifier_fnames = None
        
        #these options will be used for running a single experiment (i.e. no CV or HPT)
        self._default_classifier_opts = {k1: v1[0] for k1,v1 in self._tuning_dict['param_grid'].items()}

    def optimize_and_predict(self, out_path='.', year=''):
        """
        OLD FUNCTION - TO BE DELETED
        Get information about the learner to be used, the hyperparameters to be tuned, and the use grid-search
        """
        

        
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
        """
        OLD FUNCTION - TO BE DELETED
        Store results from tuning and prediction into json files, generating a proper tree of directories
        """
        
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
        clf = self.classifier_class(**self._default_classifier_opts)
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
            'id': xbt_ds.xbt_df['id'],
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
            xbt_year = xbt_ds.filter_obs({'year': year} )
            cats = list(xbt_ds._feature_encoders[self.target_feature].categories_)[0]
            if xbt_year.shape[0] > 0:
                X_year = xbt_year.filter_features(self.input_features).get_ml_dataset()[0]
                y_year = xbt_ds.filter_obs({'year': year} ).filter_features([self.target_feature]).get_ml_dataset()[0]

                y_res_year = clf.predict(X_year)
                prec_year, recall_year, f1_year, support_year = sklearn.metrics.precision_recall_fscore_support(
                    y_year, y_res_year, average='micro')
                prec_cat, recall_cat, f1_cat, support_cat = sklearn.metrics.precision_recall_fscore_support(
                    y_year, y_res_year)
            else:
                prec_year = 0.0
                recall_year = 0.0
                f1_year = 0.0
                support_year = 0
                prec_cat = [0.0] * len(cats)
                recall_cat = [0.0] * len(cats)
                f1_cat = [0.0] * len(cats)
                support_cat = [0.0] * len(cats)

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
        return metrics_df        
    
    def generate_prediction(self, clf, feature_name):
        if self.xbt_predictable is None:
            # checker functions check each element of the profile metadata that could be a problem. The checkers are constructed from the labelled data subset.
            checkers_labelled = {f1: c1 for f1, c1 in self.xbt_labelled.get_checkers().items() if f1 in self.input_features}   
            self.xbt_predictable = self.dataset.filter_predictable(checkers_labelled)
        
        # generate classification for predictable profiles
        res_ml1 = clf.predict(self.xbt_predictable.filter_features(self.input_features).get_ml_dataset()[0])
        res2 = list(self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_ml1).reshape(-1))        
        self.xbt_predictable.xbt_df[feature_name] = res2        
        
        def imeta_instrument(row1):
            return 'XBT: {t1[0]} ({t1[1]})'.format(t1=imeta_classification(row1))        
        
        # checking for missing values and fill in imeta
        flag_name = 'imeta_applied_{name}'.format(name=feature_name)
        self.xbt_predictable.xbt_df[flag_name] = 0
        self.xbt_predictable.xbt_df.loc[self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].index, flag_name] = 1
        self.xbt_predictable.xbt_df[flag_name] = self.xbt_predictable.xbt_df[flag_name].astype('int8')
        self.xbt_predictable.xbt_df.loc[self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].index, feature_name] = \
            self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].apply(imeta_instrument, axis=1)
        
        # merge into full dataset
        fv_dict = {feature_name: dataexploration.xbt_dataset.UNKNOWN_STR,
                   flag_name: 1
                  }
        self.dataset.merge_features(self.xbt_predictable, [feature_name, flag_name],
                               fill_values = fv_dict,
                               encoders={feature_name: self.xbt_labelled._feature_encoders[self.target_feature]},
                                output_formatters={feature_name: dataexploration.xbt_dataset.cat_output_formatter})        
        
        # fill in imeta for unpredictable values
        xbt_unknown_inputs = self.dataset.filter_obs({dataexploration.xbt_dataset.CQ_FLAG: 0})
        imeta_instrument_fallback = xbt_unknown_inputs.xbt_df.apply(imeta_instrument, axis=1)
        self.dataset.xbt_df.loc[xbt_unknown_inputs.xbt_df.index, feature_name] = imeta_instrument_fallback
        self.dataset.xbt_df[flag_name] = self.dataset.xbt_df[flag_name].astype('int8')
        
    def generate_vote_probabilities(self, result_feature_names):
        # take a list of estimators
        # generate a prediction for each
        # sum the predictions from classifiers for each class for each obs
        # generate a one hot style probability of each class based by normalising the vote counts to sum to 1 (divide by num estimators)
        vote_count = numpy.zeros([self.dataset.shape[0], len(self.dataset._feature_encoders[result_feature_names[0]].categories_[0])],dtype=numpy.float64)
        for res_name in result_feature_names:
            vote_count += self.dataset.filter_features([res_name]).get_ml_dataset()[0]
        vote_count /= float(len(result_feature_names))        
        vote_dict = {PROB_CAT_TEMPLATE.format(target=self.target_feature,
                                              clf=self.classifier_name,
                                              cat=cat1,
                                             ): vote_count[:,ix1] for ix1, cat1 in enumerate(self.dataset._feature_encoders['instrument'].categories_[0])}
        vote_dict.update({'id': self.dataset['id']})
        vote_df = pandas.DataFrame(vote_dict)        
        self.dataset.xbt_df = self.dataset.xbt_df.merge(vote_df, on='id')
        
    def export_classifiers(self):
        self.classifier_output_fnames = []
        for split_num, clf1 in self.classifiers.items():
            export_fname = CLASSIFIER_EXPORT_FNAME_TEMPLATE.format(
                split_num=split_num,
                exp=self.experiment_name,
            )
            self.classifier_output_fnames += [export_fname]
            export_path = os.path.join(self.exp_output_dir,
                                       export_fname)
            joblib.dump(clf1, export_path)
        out_dict = dict(self.json_params)
        out_dict['experiment_name'] = out_dict['experiment_name'] + '_inference'
        out_dict['classifier_fnames'] = self.classifier_output_fnames
        self.inference_out_json_path = os.path.join(self.exp_output_dir, f'xbt_param_{self.experiment_name}_inference.json')
        print(f' writing inference experiment output file to {self.inference_out_json_path}')
        with open(self.inference_out_json_path, 'w') as json_out_file:
            json.dump(out_dict, json_out_file)
