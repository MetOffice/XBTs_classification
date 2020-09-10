import argparse
import ast
import copy
import tempfile
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

import xbt.common
import dataexploration.wod
import dataexploration.xbt_dataset
from classification.imeta import imeta_classification, XBT_MAX_DEPTH


RESULT_FNAME_TEMPLATE = 'xbt_metrics_{name}.csv'
SCORE_FNAME_TEMPLATE = 'xbt_score_{name}.csv'
OUTPUT_FNAME_TEMPLATE = 'xbt_classifications_{exp_name}_{subset}.csv'
CLASSIFIER_EXPORT_FNAME_TEMPLATE = 'xbt_classifier_{exp}_{split_num}.joblib'

UNSEEN_FOLD_NAME = 'unseen_fold'
RESULT_FEATURE_TEMPLATE = '{target}_res_{clf}_split{split_num}'
PROB_CAT_TEMPLATE = '{target}_{clf}_probability_{cat}'
MAX_PROB_FEATURE_NAME = '{target}_max_prob'

OUTPUT_CQ_FLAG = 'classification_quality_flag_{var_name}'
OUTPUT_CQ_INPUT = 0
OUTPUT_CQ_ML = 1
OUTPUT_CQ_IMETA = 2


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
    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        # assign arguments
        self.data_dir = data_dir
        self.root_output_dir = output_dir
        self.output_split = output_split
        self._do_preproc_extract = do_preproc_extract
        self.json_descriptor = json_descriptor
        
        # initialise to None where appropriate
        self.dataset = None
        self.results = None
        self.classifiers = None
        self._csv_tmp_dir = None
        self._n_jobs = -1
        self._cv_output = None
        self.xbt_predictable = None
        self._exp_datestamp = None
        self.classifier_fnames = None
        self.experiment_description_dir = None
        self.score_table = None
        self._wod_encoders = {k1: enc_class1() for k1, enc_class1 in dataexploration.wod.get_wod_encoders().items()}
        
        self.ens_unseen_fraction = 0.1        
        # load experiment definition from json file
        self.primary_keys = ['learner', 'input_features', 'output_feature', 'tuning', 'split', 'experiment_name']
        self.read_json_file()
        
        if self._do_preproc_extract is not None:
            if self.preproc_params is None:
                raise RuntimeError('preprocessing parameters must be specified in the JSON '
                                   'experiment definition if preprocessing is requested '
                                   'by specify a directory for files to be preprocessed.')
        
        self.exp_output_dir = os.path.join(self.root_output_dir, self.experiment_name)
        
        
        
    def run_single_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()

        start1 = time.time()
        print('loading dataset')
        self.load_dataset()
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating splits')
        X_dict, y_dict, df_dict = self.get_train_test_unseen_sets()
            
        # fit classifier
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('training classifier')
        clf1 = self.train_classifier(X_dict['train'], y_dict['train'])
        
        # generate scores
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating metrics')
        metrics_train, score_train = self.generate_metrics(clf1, df_dict['train'], 'train', 'train')        
        metrics_test, score_test = self.generate_metrics(clf1, df_dict['test'], 'test', 'test')
        exp_results = pandas.merge(metrics_train, metrics_test, on='year')
        metrics_unseen, score_unseen = self.generate_metrics(clf1, df_dict['unseen'], 'unseen', 'unseen')
        exp_results = exp_results.merge(metrics_unseen, on='year')
        
        self.score_table = pandas.DataFrame.from_records([score_train, score_test, score_unseen])
        
        # generate imeta
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
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
            self.score_table.to_csv(
                os.path.join(self.exp_output_dir,
                             SCORE_FNAME_TEMPLATE.format(name=out_name)))
        
        # generate imeta algorithm results for the whole dataset
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
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
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            self.dataset.output_data(
                self.exp_output_dir,
                fname_template=OUTPUT_FNAME_TEMPLATE,
                exp_name=out_name,
                output_split=self.output_split,
                target_features=[feature_name],
            )
            
        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()
            
        return (self.results, self.classifiers)
    
    def _do_ensemble_experiment(self, classifier_obj, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()
        
        start1 = time.time()
        print('loading dataset')
        self.load_dataset()
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        
        
        # get train/test/unseen sets
        print('generating splits')
        # using this function to ensure an even split by year seems to cause a dramatiuc degradation of classification results, so turning this off for now,
        # I suspect there is a bug in the function that is doing something weird to the split
        # ensemble_unseen_cruise_numbers = self.xbt_labelled.sample_feature_values(self.unseen_feature, fraction=self.ens_unseen_fraction, split_feature='year')

        ensemble_unseen_cruise_numbers = self.xbt_labelled.sample_feature_values(self.unseen_feature, fraction=self.ens_unseen_fraction)
        xbt_ens_unseen = self.xbt_labelled.filter_obs({self.unseen_feature: ensemble_unseen_cruise_numbers}, mode='include', check_type='in_filter_set')
        xbt_ens_working = self.xbt_labelled.filter_obs({self.unseen_feature: ensemble_unseen_cruise_numbers}, mode='exclude', check_type='in_filter_set')        
        
        X_ens_working = xbt_ens_working.filter_features(self.input_features).get_ml_dataset()[0]
        y_ens_working = xbt_ens_working.filter_features([self.target_feature]).get_ml_dataset()[0]
        
        xbt_ens_working.generate_folds_by_feature(self.unseen_feature, self.num_unseen_splits, UNSEEN_FOLD_NAME)
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        group_cv1 = sklearn.model_selection.GroupKFold(n_splits=self.num_unseen_splits)
        # now run the outer cross-validation with the grid search HPT as the classifier,
        # so for each split, HPT will be run with CV on each set of hyperparameters
        print('running cross validation')
        scores = sklearn.model_selection.cross_validate(
            classifier_obj,
            X_ens_working, y_ens_working, 
            groups=xbt_ens_working[UNSEEN_FOLD_NAME], 
            cv=group_cv1,
            return_estimator=self.return_estimator,
            return_train_score=self.return_train_score,
            scoring=self._tuning_dict['cv_metrics'],
            n_jobs=self._n_jobs,
        )        

        
        self._cv_output = scores
        self.classifiers = {split_num: est1 
                            for split_num, est1 in enumerate(scores['estimator'])}
       
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating probabilities for evaluation.')
        
        (df_ens_working, 
         df_ens_unseen, 
         metrics_df_ens,
        ens_scores_list) =self._evaluate_vote_probs(xbt_ens_working,
                                                    xbt_ens_unseen,
                                                    scores,
                                                   )
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('calculating metrics')
        metrics_list = {}
        scores_list = ens_scores_list
        for split_num, estimator in self.classifiers.items():
            xbt_train = xbt_ens_working.filter_obs({UNSEEN_FOLD_NAME: split_num}, mode='exclude')
            xbt_test = xbt_ens_working.filter_obs({UNSEEN_FOLD_NAME: split_num}, mode='include')
            train_metrics, train_scores = self.generate_metrics(estimator, xbt_train, 'train_{0}'.format(split_num), 'train')
            test_metrics, test_scores = self.generate_metrics(estimator, xbt_test, 'test_{0}'.format(split_num), 'test')
            unseen_metrics, unseen_scores = self.generate_metrics(estimator, xbt_ens_unseen, 'unseen_{0}'.format(split_num), 'unseen' )            
            metrics_list[split_num] = pandas.merge(train_metrics, test_metrics, on='year')   
            scores_list += [train_scores, test_scores, unseen_scores]

        self.score_table = pandas.DataFrame.from_records(scores_list)
        print('overall scores for trained classifiers:')
        print(self.score_table)
            
        metrics_df_merge = None
        for label1, metrics1  in metrics_list.items():
            if metrics_df_merge is None:
                metrics_df_merge = metrics1
            else:
                metrics_df_merge = pandas.merge(metrics_df_merge, metrics1, on='year')     
        metrics_df_merge = pandas.merge(metrics_df_merge, metrics_df_ens, on='year')
        self.results = metrics_df_merge
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        if write_results:
            print('output results to a file')
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            self.results.to_csv(
                os.path.join(self.exp_output_dir,
                             RESULT_FNAME_TEMPLATE.format(name=out_name)))
            self.score_table.to_csv(
                os.path.join(self.exp_output_dir,
                             SCORE_FNAME_TEMPLATE.format(name=out_name)))
        
        
        # generate imeta algorithm results for the whole dataset
        imeta_df = self.generate_imeta(self.dataset)
        imeta_df = imeta_df.rename(
            columns={'instrument': 'imeta_instrument',
                     'model': 'imeta_model',
                     'manufacturer': 'imeta_manufacturer',
            })
        self.dataset.xbt_df = self.dataset.xbt_df.merge(imeta_df[['id', 'imeta_{0}'.format(self.target_feature)]])
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print(' run prediction on full dataset')
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
        
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('output predictions')
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            self.dataset.output_data(
                self.exp_output_dir,
                fname_template=OUTPUT_FNAME_TEMPLATE,
                exp_name=out_name,
                output_split=self.output_split,
                target_features=[],
            )
                    
        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()
            
        return (self.results, self.classifiers)
    
    def run_cv_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        print('using classifier opts:\n' + str(self._default_classifier_opts))
        classifier_obj = self.classifier_class(**self._default_classifier_opts)
        return self._do_ensemble_experiment(classifier_obj, 
                                      write_results, 
                                      write_predictions, 
                                      export_classifiers)
        
    def run_cvhpt_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        # create objects for cross validation and hyperparameter tuning
        # first set up objects for the inner cross validation, which run for each
        # set of hyperparameters in the grid search.
        classifier_obj = self.classifier_class(**self._default_classifier_opts)
        grid_search_cv = sklearn.model_selection.GridSearchCV(
            classifier_obj,
            param_grid = self._tuning_dict['param_grid'],  
            scoring=self._tuning_dict['scoring'],
            cv=self.num_training_splits,
        )
        return self._do_ensemble_experiment(grid_search_cv, 
                                      write_results, 
                                      write_predictions, 
                                      export_classifiers)
    
    
    def run_inference(self, write_predictions=True):
        """
        """
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()
        
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
            self.dataset.output_data(
                self.exp_output_dir,
                fname_template=OUTPUT_FNAME_TEMPLATE,
                exp_name=out_name,
                output_split=self.output_split,
                target_features=[],
            )
                    
        return self.classifiers
    
    
    def _construct_dataset_obj(self):
        if self._do_preproc_extract:
            self._csv_tmp_dir = tempfile.TemporaryDirectory()
            self.dataset = dataexploration.xbt_dataset.XbtDataset(
                self._csv_tmp_dir, 
                self.year_range, 
                nc_dir=self.data_dir,
                pp_prefix=self.preproc_params['prefix'],
                pp_suffix=self.preproc_params['suffix'],
            )
        else:
            self.dataset = dataexploration.xbt_dataset.XbtDataset(
                self.data_dir, 
                self.year_range, 
            )
    
    def load_dataset(self):
        """
        create a XBTDataset
        only load the specified input and target features, taken from the parameters JSON file
        """
        # the actual construction is put into a sperate function, so child 
        # classes for different platforms can handle any platform specific stuff 
        self._construct_dataset_obj()
        
        # get the year range from the data once it has loaded if it was not specified previously
        if self.year_range is None:
            self.year_range = self.dataset.year_range
            
        self.xbt_labelled = self.dataset.filter_obs({'labelled': 'labelled'})
        
        # initialise the feature encoders on the labelled data
        _ = self.xbt_labelled.get_ml_dataset(return_data=False)
        _ = self.xbt_labelled.filter_features(dataexploration.xbt_dataset.TARGET_FEATURES).encode_target(return_data=False)

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
        try: 
            self.year_range = (self.json_params['year_range'][0], self.json_params['year_range'][1])
        except:
            self.year_range = None # if not set, this will be derived from the data.
            
        self.input_features = self.json_params['input_features']
        self.target_feature = self.json_params['output_feature']
        
        self.num_training_splits = int(self.json_params['split']['num_training_splits'])
        self.test_fraction = 1.0 / self.num_training_splits
        self.train_fraction = 1.0 - self.test_fraction
        self.num_unseen_splits = int(self.json_params['split']['num_unseen_splits'])
        self.unseen_fraction = 1.0 / self.num_unseen_splits
        self.unseen_feature = self.json_params['split']['unseen_feature']
        self.balance_features = self.json_params['split']['balance_features']
        
        try:
            self.preproc_params = self.json_params['preproc']
        except KeyError:
            self.preproc_params = None            
        
        
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

            cats = list(xbt_ds._feature_encoders[self.target_feature].classes_)
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
        
    def generate_metrics(self, clf, xbt_ds, data_label, split_subset):
        metric_list = []
        for year in range(self.year_range[0],self.year_range[1]):
            xbt_year = xbt_ds.filter_obs({'year': year} )
            cats = list(xbt_ds._feature_encoders[self.target_feature].classes_)
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
                           column_template.format(data=data_label, metric='support', subset='all'): support_year,
                          }

            metric_dict.update({column_template.format(data=data_label, metric='precision', subset=cat): val for cat, val in zip(cats, prec_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='recall', subset=cat): val for cat, val in zip(cats, recall_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='f1', subset=cat): val for cat, val in zip(cats, f1_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='support', subset=cat): val for cat, val in zip(cats, support_cat)})
            metric_list += [metric_dict]
        metrics_df = pandas.DataFrame.from_records(metric_list)
            
        X_full = xbt_ds.filter_features(self.input_features).get_ml_dataset()[0]
        y_full = xbt_ds.filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_full = clf.predict(X_full)
        prec_full, recall_full, f1_full, support_full = sklearn.metrics.precision_recall_fscore_support(
                y_full, y_res_full, average='micro')
        metrics_full = {
            'name': data_label,
            'subset': split_subset, 
            'precision_all': prec_full,
            'recall_all': recall_full,
            'f1_all': f1_full,
            'support_all': support_full,
            }
        return metrics_df, metrics_full       
    
    def generate_prediction(self, clf, feature_name):
        if self.xbt_predictable is None:
            # checker functions check each element of the profile metadata that could be a problem. The checkers are constructed from the labelled data subset.
            checkers_labelled = {f1: c1 for f1, c1 in self.xbt_labelled.get_checkers().items() if f1 in self.input_features}   
            self.xbt_predictable = self.dataset.filter_predictable(checkers_labelled)
        
        # generate classification for predictable profiles
        res_ml1 = clf.predict(self.xbt_predictable.filter_features(self.input_features).get_ml_dataset()[0])
        res2 = list(self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_ml1).reshape(-1))        
        # use known instrument type label for the labelled data, so overwrite the predictions with the known values where we have them
        self.xbt_predictable.xbt_df.loc[self.xbt_labelled.xbt_df.index, feature_name] = self.xbt_predictable.xbt_df.loc[self.xbt_labelled.xbt_df.index, self.target_feature]
        self.xbt_predictable.xbt_df[feature_name] = res2        
        
        def imeta_instrument(row1):
            return 'XBT: {t1[0]} ({t1[1]})'.format(t1=imeta_classification(row1))        

        # checking for missing values and fill in imeta
        flag_name = OUTPUT_CQ_FLAG.format(var_name=feature_name)
        self.xbt_predictable.xbt_df[flag_name] = OUTPUT_CQ_ML 
        self.xbt_predictable.xbt_df.loc[self.xbt_labelled.xbt_df.index, flag_name] = OUTPUT_CQ_INPUT
        self.xbt_predictable.xbt_df.loc[self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].index, flag_name] = OUTPUT_CQ_IMETA
        
        
        self.xbt_predictable.xbt_df[flag_name] = self.xbt_predictable.xbt_df[flag_name].astype('int8')
        self.xbt_predictable.xbt_df.loc[self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].index, feature_name] = \
            self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].apply(imeta_instrument, axis=1)
        
        # merge into full dataset
        # first, define what to use for missing values when merging
        fv_dict = {feature_name: dataexploration.xbt_dataset.UNKNOWN_STR,
                   flag_name: OUTPUT_CQ_IMETA 
                  }
        self.dataset.merge_features(self.xbt_predictable, [feature_name, flag_name],
                               fill_values = fv_dict,
                               feature_encoders={feature_name: self.xbt_labelled._feature_encoders[self.target_feature]},
                               target_encoders={feature_name: self.xbt_labelled._target_encoders[self.target_feature]},
                                output_formatters={feature_name: dataexploration.xbt_dataset.cat_output_formatter})        
        
        # fill in imeta for unpredictable values
        xbt_unknown_inputs = self.dataset.filter_obs({dataexploration.xbt_dataset.PREDICTABLE_FLAG: 0})
        imeta_instrument_fallback = xbt_unknown_inputs.xbt_df.apply(imeta_instrument, axis=1)
        self.dataset.xbt_df.loc[xbt_unknown_inputs.xbt_df.index, feature_name] = imeta_instrument_fallback
        self.dataset.xbt_df[flag_name] = self.dataset.xbt_df[flag_name].astype('int8')
        
        # add WOD code version of output
        coded_feature_name = feature_name + '_code'
        try:
            wod_target_encoder = self._wod_encoders[self.target_feature]
            self.dataset.xbt_df[coded_feature_name] = self.dataset.xbt_df[feature_name].apply(wod_target_encoder.name_to_code)
        except KeyError:
            print(f'No WOD encoder for target feature {self.target_feature}, encoded version of data not produced.')
        
    def generate_vote_probabilities(self, result_feature_names):
        # take a list of estimators
        # generate a prediction for each
        # sum the predictions from classifiers for each class for each obs
        # generate a one hot style probability of each class based by normalising the vote counts to sum to 1 (divide by num estimators)
        vote_count = numpy.zeros([self.dataset.shape[0], len(self.dataset._feature_encoders[result_feature_names[0]].classes_)],dtype=numpy.float64)
        for res_name in result_feature_names:
            vote_count += self.dataset.filter_features([res_name]).encode_target()[0]
        vote_count /= float(len(result_feature_names))        
        
        res_full_ensemble = vote_count.argmax(axis=1)
        instr_res_full_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_full_ensemble)        
        
        vote_dict = {PROB_CAT_TEMPLATE.format(target=self.target_feature,
                                              clf=self.classifier_name,
                                              cat=cat1,
                                             ): vote_count[:,ix1] for ix1, cat1 in enumerate(self.dataset._feature_encoders['instrument'].classes_)}
        vote_dict.update({'id': self.dataset['id'],
                          MAX_PROB_FEATURE_NAME.format(target=self.target_feature): instr_res_full_ensemble,
                         })
        vote_df = pandas.DataFrame(vote_dict)        
        self.dataset.xbt_df = self.dataset.xbt_df.merge(vote_df, on='id')
        

    def _evaluate_vote_probs(self, xbt_ens_working, xbt_ens_unseen, scores):
        res_ens_working = {'id': xbt_ens_working.xbt_df['id']}
        res_ens_unseen = {'id': xbt_ens_unseen.xbt_df['id']}
        vote_count_working = numpy.zeros([xbt_ens_working.shape[0], self.xbt_labelled._feature_encoders[self.target_feature].classes_.shape[0]],dtype=numpy.float64)
        vote_count_unseen = numpy.zeros([xbt_ens_unseen.shape[0], self.xbt_labelled._feature_encoders[self.target_feature].classes_.shape[0]],dtype=numpy.float64)
        result_feature_names = []
        # classifications_df = None
        for split_num, estimator in enumerate(scores['estimator']):
            res_name = RESULT_FEATURE_TEMPLATE.format(
                target=self.target_feature,
                clf=self.classifier_name,
                split_num=split_num)
            result_feature_names += [res_name]
            res_ml1_working = estimator.predict(xbt_ens_working.filter_features(self.input_features).get_ml_dataset()[0])
            res2_working = xbt_ens_working._feature_encoders[self.target_feature].inverse_transform(res_ml1_working).reshape(-1,1)
            res_ens_working[res_name] = res2_working.reshape(-1)
            vote_count_working += xbt_ens_working._target_encoders[self.target_feature].transform(res2_working)
    
            res_ml1_unseen = estimator.predict(xbt_ens_unseen.filter_features(self.input_features).get_ml_dataset()[0])
            res2_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].inverse_transform(res_ml1_unseen).reshape(-1,1)
            res_ens_unseen[res_name] = res2_unseen.reshape(-1)
            vote_count_unseen += xbt_ens_unseen._target_encoders[self.target_feature].transform(res2_unseen)
    
        df_ens_working = pandas.DataFrame(res_ens_working)
        df_ens_unseen = pandas.DataFrame(res_ens_unseen)

        vote_count_working /= float(len(res_ens_working.keys()))    
        vote_count_unseen /= float(len(res_ens_working.keys()))         
        
        max_prob_feature_name = f'{self.target_feature}_max_prob'
        res_working_ensemble = vote_count_working.argmax(axis=1)
        instr_res_working_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_working_ensemble)
        df_ens_working[max_prob_feature_name] = instr_res_working_ensemble
        res_unseen_ensemble = vote_count_unseen.argmax(axis=1)
        instr_res_unseen_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_unseen_ensemble)
        df_ens_unseen[max_prob_feature_name] = instr_res_unseen_ensemble
        
        df_ens_working = pandas.merge(df_ens_working, xbt_ens_working.xbt_df[['id', 'year']])
        df_ens_unseen = pandas.merge(df_ens_unseen, xbt_ens_unseen.xbt_df[['id', 'year']])        

        metric_list_ens = []
        for year in range(self.year_range[0],self.year_range[1]):
            y_ens_working = xbt_ens_working.filter_obs({'year': year} ).filter_features([self.target_feature]).get_ml_dataset()[0]
            y_ens_unseen = xbt_ens_unseen.filter_obs({'year': year} ).filter_features([self.target_feature]).get_ml_dataset()[0]

            y_res_working = xbt_ens_working._feature_encoders[self.target_feature].transform( df_ens_working[df_ens_working.year == year][max_prob_feature_name])    
            y_res_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].transform( df_ens_unseen[df_ens_unseen.year == year][max_prob_feature_name])    
            cats = list(xbt_ens_working._feature_encoders[self.target_feature].classes_)
            prec_ens_working, recall_ens_working, f1_ens_working, support_ens_working = sklearn.metrics.precision_recall_fscore_support(
                y_ens_working, y_res_working, average='micro', labels=range(0,len(cats)))
            prec_ens_unseen, recall_ens_unseen, f1_ens_unseen, support_ens_unseen = sklearn.metrics.precision_recall_fscore_support(
                y_ens_unseen, y_res_unseen, average='micro', labels=range(0,len(cats)))
            column_template = '{metric}_{data}_{subset}'
            metric_dict = {'year': year,
                       column_template.format(data='ens_working', metric='precision', subset='all'): prec_ens_working,
                       column_template.format(data='ens_working', metric='recall', subset='all'): recall_ens_working,
                       column_template.format(data='ens_working', metric='f1', subset='all'): f1_ens_working,
                       column_template.format(data='ens_working', metric='support', subset='all'):     support_ens_working,
                       column_template.format(data='ens_unseen', metric='precision', subset='all'): prec_ens_unseen,
                       column_template.format(data='ens_unseen', metric='recall', subset='all'): recall_ens_unseen,
                       column_template.format(data='ens_unseen', metric='f1', subset='all'): f1_ens_unseen,
                       column_template.format(data='ens_unseen', metric='support', subset='all'):     support_ens_unseen,
                           }

            metric_list_ens += [metric_dict]
        metrics_df_ens = pandas.DataFrame.from_records(metric_list_ens)            
        
        # calculate scores for the ensemble classifier output
        cats = list(xbt_ens_working._feature_encoders[self.target_feature].classes_)
        y_ens_working = xbt_ens_working.filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_working = xbt_ens_working._feature_encoders[self.target_feature].transform( df_ens_working[max_prob_feature_name])    
        prec_ens_working, recall_ens_working, f1_ens_working, support_ens_working = sklearn.metrics.precision_recall_fscore_support(
            y_ens_working, y_res_working, average='micro', labels=range(0,len(cats)))

        column_template = '{metric}_{data}_{subset}'
        scores_ens_working = {
            'name': 'ens_working',
            'subset': 'train',
            'precision_all': prec_ens_working,
            'recall_all': recall_ens_working,
            'f1_all': f1_ens_working,
            'support_all': support_ens_working,
        }
        y_ens_unseen = xbt_ens_unseen.filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].transform( df_ens_unseen[max_prob_feature_name])    
        prec_ens_unseen, recall_ens_unseen, f1_ens_unseen, support_ens_unseen = sklearn.metrics.precision_recall_fscore_support(
            y_ens_unseen, y_res_unseen, average='micro', labels=range(0,len(cats)))
        scores_ens_unseen = {
            'name': 'ens_unseen',
            'subset': 'unseen',
            'precision_all': prec_ens_unseen,
            'recall_all': recall_ens_unseen,
            'f1_all': f1_ens_unseen,
            'support_all': f1_ens_unseen,
        }
            
        return (df_ens_working,
                df_ens_unseen,
                metrics_df_ens,
                [scores_ens_working, scores_ens_unseen],
               )
        
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

               

    
    