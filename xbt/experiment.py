import ast
import datetime
import tempfile
import importlib
import joblib
import json
import numpy
import os
import pandas
import time
import abc
import random
import itertools
import functools

import sklearn.metrics
import sklearn.inspection

pandas.options.mode.chained_assignment = None

import xbt.common
import xbt.wod
import xbt.dataset
from xbt.imeta import imeta_classification

IMPORTANCE_REPEATS = 20

RESULT_FNAME_TEMPLATE = 'xbt_metrics_{name}.csv'
IMPORTANCE_FNAME_TEMPLATE = 'xbt_importance_{name}.csv'
SCORE_FNAME_TEMPLATE = 'xbt_score_{name}.csv'
PARAM_FNAME_TEMPLATE = 'xbt_hyperparams_{name}.json'
OUTPUT_FNAME_TEMPLATE = 'xbt_classifications_{exp_name}_{subset}.csv'
CLASSIFIER_EXPORT_FNAME_TEMPLATE = 'xbt_classifier_{exp}_{split_num}.joblib'

UNSEEN_FOLD_NAME = 'unseen_fold'
RESULT_FEATURE_TEMPLATE = '{target}_res_{clf}_split{split_num}'
PROB_CAT_TEMPLATE = '{target}_{clf}_probability_{cat}'
MAX_PROB_FEATURE_NAME = 'res_{target}_max_prob'
RESAMPLE_FEATURE_TEMPLATE = 'resample_train_{resample_index}'

TEST_VAR_NAME = 'test'
TEST_PART_NAME = 'test_part'
TEST_WHOLE_NAME = 'test_whole'
VALIDATION_VAR_NAME = 'validation'
TRAIN_VAR_NAME = 'train'

METRIC_SET_ALL = 'all';
METRIC_SET_PER_YEAR = 'per_year'
METRIC_SET_PER_CLASS = 'per_class'

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


class ClassificationExperiment(abc.ABC):
    """
    Class designed for implementing features engineering, design of the input space, algorithms fine tuning and
    delivering outut prediction
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
        self._wod_encoders = {k1: enc_class1() for k1, enc_class1 in xbt.wod.get_wod_encoders().items()}

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
        self.metrics_out_path = None
        self.scores_out_path = None
        self.predictions_out_path_list = []
        self._random_state = random.randint(1, 2**32)

    @abc.abstractmethod
    def run_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        print('running run_experiment method in ClassificationExperiment base class.')

    def _construct_dataset_obj(self):
        if self._do_preproc_extract:
            self._csv_tmp_dir_obj = tempfile.TemporaryDirectory()
            self._csv_tmp_dir = self._csv_tmp_dir_obj.name
            self.dataset = xbt.dataset.XbtDataset(
                directory=self.data_dir,
                year_range=self.year_range,
                do_preproc_extract=self._do_preproc_extract,
                pp_csv_dir=self._csv_tmp_dir,
                pp_prefix=self.preproc_params['prefix'],
                pp_suffix=self.preproc_params['suffix'],
            )
        else:
            self.dataset = xbt.dataset.XbtDataset(
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
        _ = self.xbt_labelled.filter_features(xbt.dataset.TARGET_FEATURES).encode_target(return_data=False)
        self.instrument_list = list(
            self.xbt_labelled._feature_encoders[self.target_feature].classes_)

    def setup_metrics(self):

        self.do_avg_args_dict = {'labels': list(range(0, len(self.instrument_list))),
                                 'average': 'weighted'}

        self.metrics_defs_dict = {
            'recall': {'metric_func': sklearn.metrics.recall_score,
                       'metric_args_dict': self.do_avg_args_dict},
            'precision': {'metric_func': sklearn.metrics.precision_score,
                          'metric_args_dict': self.do_avg_args_dict},
            'accuracy': {'metric_func': sklearn.metrics.accuracy_score,
                         'metric_args_dict': {}},
            'f1': {'metric_func': sklearn.metrics.f1_score,
                   'metric_args_dict': self.do_avg_args_dict},
            'balanced_accuracy': {
                'metric_func': sklearn.metrics.balanced_accuracy_score,
                'metric_args_dict': {}},
        }

        self.results = {
            TEST_VAR_NAME: {},
            TEST_WHOLE_NAME: {},
            TEST_PART_NAME: {},
            TRAIN_VAR_NAME: {},
            VALIDATION_VAR_NAME: {},
        }


    def _check_output_dir(self):
        if not self.exp_output_dir:
            raise RuntimeError(f'experiment output directory path ({self.exp_output_dir}) not defined.')

        if not os.path.isdir(self.exp_output_dir):
            os.makedirs(self.exp_output_dir)

    def read_json_file(self):
        """Open json descriptor file, load its content into a dictionary"""

        print(f'reading JSON experiment definition from {self.json_descriptor}')
        if not os.path.isfile(self.json_descriptor):
            raise ValueError(f'Missing json descriptor {self.json_descriptor}!')
        if not os.path.isabs(self.json_descriptor):
            self.json_descriptor = os.path.abspath(self.json_descriptor)
        self.experiment_description_dir, self.json_fname = os.path.split(self.json_descriptor)
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
            self.year_range = None  # if not set, this will be derived from the data.

        self.input_features = self.json_params['input_features']
        self.target_feature = self.json_params['output_feature']

        self.num_training_splits = int(self.json_params['split']['num_training_splits'])
        self.test_fraction = 1.0 / self.num_training_splits
        self.train_fraction = 1.0 - self.test_fraction
        self.num_test_splits = int(self.json_params['split']['num_test_splits'])
        self.unseen_fraction = 1.0 / self.num_test_splits
        self.unseen_feature = self.json_params['split']['unseen_feature']
        self.balance_features = self.json_params['split']['balance_features']

        try:
            self.preproc_params = self.json_params['preproc']
        except KeyError:
            self.preproc_params = None

        self.return_estimator = self.json_params['tuning']['return_estimator']
        self.return_train_score = self.json_params['tuning']['return_train_score']

        learner_dictionary = self.json_params['learner']
        learner_module_name = learner_dictionary['module_name']
        learner_class_name = learner_dictionary['python_class']
        learner_module = importlib.import_module(learner_module_name)
        learner_class = getattr(learner_module, learner_class_name)
        self.classifier_name = learner_dictionary['name']
        self.classifier_class = learner_class
        self._tuning_dict = self.json_params['tuning']
        try:
            self.classifier_fnames = self.json_params['classifier_fnames']
        except KeyError:
            self.classifier_fnames = None

        # these options will be used for running a single experiment (i.e. no CV or HPT)
        self._default_classifier_opts = {k1: v1[0] for k1, v1 in self._tuning_dict['param_grid'].items()}

    def generate_test_working_datasets(self):

        test_cruise_numbers = self.xbt_labelled.sample_feature_values(
            'cruise_number', fraction=0.1)
        test_indices = list(itertools.chain.from_iterable([list(
            self.xbt_labelled.filter_obs(
                {self.target_feature: selected_instrument}).xbt_df.sample(
                frac=0.1).index)
                                                           for
                                                           selected_instrument
                                                           in self.xbt_labelled[
                                                               'instrument'].unique()]))

        self.xbt_labelled.xbt_df[TEST_VAR_NAME] = self.xbt_labelled.xbt_df[
            'cruise_number'].isin(test_cruise_numbers)
        self.xbt_labelled.xbt_df[TEST_WHOLE_NAME] = self.xbt_labelled.xbt_df[
            'cruise_number'].isin(test_cruise_numbers)

        # label test data where part of the cruise is in the train/validation sets
        self.xbt_labelled.xbt_df.loc[test_indices, TEST_VAR_NAME] = True
        self.xbt_labelled.xbt_df[TEST_PART_NAME] = False
        self.xbt_labelled.xbt_df.loc[test_indices, TEST_PART_NAME] = True

        self.xbt_test = self.xbt_labelled.filter_obs({TEST_VAR_NAME: True})
        self.xbt_working = self.xbt_labelled.filter_obs({TEST_VAR_NAME: False})
        self.xbt_test_whole =self. xbt_labelled.filter_obs({TEST_WHOLE_NAME: True})
        self.xbt_test_part = self.xbt_labelled.filter_obs({TEST_PART_NAME: True})

    def get_train_test_unseen_sets(self):
        unseen_cruise_numbers = self.xbt_labelled.sample_feature_values(self.unseen_feature,
                                                                        fraction=self.unseen_fraction)
        if self.unseen_feature:
            xbt_unseen = self.xbt_labelled.filter_obs({self.unseen_feature: unseen_cruise_numbers}, mode='include',
                                                      check_type='in_filter_set')
            xbt_working = self.xbt_labelled.filter_obs({self.unseen_feature: unseen_cruise_numbers}, mode='exclude',
                                                       check_type='in_filter_set')
        else:
            xbt_unseen = None
            xbt_working = self.xbt_labelled
        xbt_train_all, xbt_test_all = xbt_working.train_test_split(refresh=True,
                                                                   features=self.balance_features,
                                                                   train_fraction=self.train_fraction)
        X_train_all,_,_,_, feature_names = xbt_train_all.filter_features(self.input_features).get_ml_dataset()
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
                   'unseen': xbt_unseen, }
        X_dict = {'train': X_train_all,
                  'test': X_test_all,
                  'unseen': X_unseen_all,
                  }
        y_dict = {'train': y_train_all,
                  'test': y_test_all,
                  'unseen': y_unseen_all,
                  }
        return X_dict, y_dict, df_dict, feature_names

    def predict_and_calc_metric_subset(self, xbt_subset, clf1, filter_dict, metric_func, metric_args_dict):
        if filter_dict:
            xbt_selected = xbt_subset.filter_obs(filter_dict)
        else:
            xbt_selected = xbt_subset
        if xbt_selected.shape[0] == 0:
            return 0.0
        metrics_result = metric_func(
            clf1.predict(
                xbt_selected.filter_features(self.input_features).get_ml_dataset()[0]),
            xbt_selected.filter_features([self.target_feature]).get_ml_dataset()[0],
            **metric_args_dict)
        return metrics_result

    def calc_column_metric_subset(self, xbt_subset, column_name, filter_dict, metric_func, metric_args_dict):
        if filter_dict:
            xbt_selected = xbt_subset.filter_obs(filter_dict)
        else:
            xbt_selected = xbt_subset
        if xbt_selected.shape[0] == 0:
            return 0.0
        metrics_result = metric_func(
            xbt_selected.filter_features([column_name]).get_ml_dataset()[0],
            xbt_selected.filter_features([self.target_feature]).get_ml_dataset()[0],
            **metric_args_dict)
        return metrics_result

    def score_year(self, xbt_df, year, clf):
        X_year = xbt_df.filter_obs({'year': year}, ).filter_features(self.input_features).get_ml_dataset()[0]
        y_year = xbt_df.filter_obs({'year': year}).filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_year = clf.predict(X_year)
        metric_year = sklearn.metrics.precision_recall_fscore_support(
            y_year, y_res_year, average='micro')
        return metric_year

    def generate_imeta(self, xbt_ds):
        try:
            if not self.imeta_feature_name:
                self.imeta_feature_name = f'imeta_{self.target_feature}'
        except:
            self.imeta_feature_name = f'imeta_{self.target_feature}'
        imeta_classes = xbt_ds.xbt_df.apply(imeta_classification, axis=1)
        imeta_df = pandas.DataFrame.from_dict({
            'id': xbt_ds.xbt_df['id'],
            self.imeta_feature_name: imeta_classes.apply(lambda t1: f'XBT: {t1[0]} ({t1[1]})'),
        })
        return imeta_df

    def score_imeta(self, xbt_ds, data_label):
        imeta_scores = []
        for year in range(self.year_range[0], self.year_range[1]):
            xbt_year = xbt_ds.filter_obs({'year': year})
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
            metric_dict.update(
                {'precision_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, prec_cat)})
            metric_dict.update(
                {'recall_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, recall_cat)})
            metric_dict.update(
                {'f1_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, f1_cat)})
            metric_dict.update(
                {'support_imeta_{dl}_{cat}'.format(cat=cat, dl=data_label): val for cat, val in zip(cats, support_cat)})
            imeta_scores += [metric_dict]

        return pandas.DataFrame.from_records(imeta_scores)

    def generate_metrics(self, clf, xbt_ds, data_label, split_subset):
        metric_list = []
        for year in range(self.year_range[0], self.year_range[1]):
            xbt_year = xbt_ds.filter_obs({'year': year})
            cats = list(xbt_ds._feature_encoders[self.target_feature].classes_)
            if xbt_year.shape[0] > 0:
                X_year = xbt_year.filter_features(self.input_features).get_ml_dataset()[0]
                y_year = xbt_ds.filter_obs({'year': year}).filter_features([self.target_feature]).get_ml_dataset()[0]

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

            metric_dict.update(
                {column_template.format(data=data_label, metric='precision', subset=cat): val for cat, val in
                 zip(cats, prec_cat)})
            metric_dict.update(
                {column_template.format(data=data_label, metric='recall', subset=cat): val for cat, val in
                 zip(cats, recall_cat)})
            metric_dict.update({column_template.format(data=data_label, metric='f1', subset=cat): val for cat, val in
                                zip(cats, f1_cat)})
            metric_dict.update(
                {column_template.format(data=data_label, metric='support', subset=cat): val for cat, val in
                 zip(cats, support_cat)})
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

    def generate_predictable_subset(self):
        if self.xbt_predictable is None:
            checkers_labelled = {f1: c1 for f1, c1 in
                                 self.xbt_labelled.get_checkers().items() if
                                 f1 in self.input_features}

            self.xbt_predictable = self.dataset.filter_predictable(
                checkers_labelled)
            self.xbt_predictable._feature_encoders = self.xbt_labelled._feature_encoders
            self.xbt_predictable._target_encoders = self.xbt_labelled._target_encoders

    def generate_prediction(self, clf, feature_name):
        self.generate_predictable_subset()

        # generate classification for predictable profiles
        res_ml1 = clf.predict(self.xbt_predictable.filter_features(self.input_features).get_ml_dataset()[0])
        res2 = list(self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(res_ml1).reshape(-1))
        # use known instrument type label for the labelled data, so overwrite the predictions with the known values
        # where we have them
        self.xbt_predictable.xbt_df.loc[self.xbt_labelled.xbt_df.index, feature_name] = self.xbt_predictable.xbt_df.loc[
            self.xbt_labelled.xbt_df.index, self.target_feature]
        self.xbt_predictable.xbt_df[feature_name] = res2

        def imeta_instrument(row1):
            return 'XBT: {t1[0]} ({t1[1]})'.format(t1=imeta_classification(row1))

            # checking for missing values and fill in imeta

        flag_name = OUTPUT_CQ_FLAG.format(var_name=feature_name)
        self.xbt_predictable.xbt_df[flag_name] = OUTPUT_CQ_ML
        self.xbt_predictable.xbt_df.loc[self.xbt_labelled.xbt_df.index, flag_name] = OUTPUT_CQ_INPUT
        self.xbt_predictable.xbt_df.loc[self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[
            feature_name].isnull()].index, flag_name] = OUTPUT_CQ_IMETA

        self.xbt_predictable.xbt_df[flag_name] = self.xbt_predictable.xbt_df[flag_name].astype('int8')
        self.xbt_predictable.xbt_df.loc[
            self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].index, feature_name] = \
            self.xbt_predictable.xbt_df[self.xbt_predictable.xbt_df[feature_name].isnull()].apply(imeta_instrument,
                                                                                                  axis=1)

        # merge into full dataset
        # first, define what to use for missing values when merging
        fv_dict = {feature_name: xbt.dataset.UNKNOWN_STR,
                   flag_name: OUTPUT_CQ_IMETA
                   }
        self.dataset.merge_features(self.xbt_predictable, [feature_name, flag_name],
                                    fill_values=fv_dict,
                                    feature_encoders={
                                        feature_name: self.xbt_labelled._feature_encoders[self.target_feature]},
                                    target_encoders={
                                        feature_name: self.xbt_labelled._target_encoders[self.target_feature]},
                                    output_formatters={feature_name: [xbt.dataset.cat_output_formatter]})

        # fill in imeta for unpredictable values
        xbt_unknown_inputs = self.dataset.filter_obs({xbt.dataset.PREDICTABLE_FLAG: 0})
        imeta_instrument_fallback = xbt_unknown_inputs.xbt_df.apply(imeta_instrument, axis=1)
        self.dataset.xbt_df.loc[xbt_unknown_inputs.xbt_df.index, feature_name] = imeta_instrument_fallback
        self.dataset.xbt_df[flag_name] = self.dataset.xbt_df[flag_name].astype('int8')

        # add WOD code version of output
        coded_feature_name = feature_name + '_code'
        try:
            wod_target_encoder = self._wod_encoders[self.target_feature]
            self.dataset.xbt_df[coded_feature_name] = self.dataset.xbt_df[feature_name].apply(
                wod_target_encoder.name_to_code)
        except KeyError:
            print(f'No WOD encoder for target feature {self.target_feature}, encoded version of data not produced.')

    def generate_vote_probabilities(self, result_feature_names):
        # take a list of estimators
        # generate a prediction for each
        # sum the predictions from classifiers for each class for each obs
        # generate a one hot style probability of each class based by normalising the vote counts to sum to 1 (divide
        # by num estimators)
        vote_count = numpy.zeros(
            [self.dataset.shape[0], len(self.dataset._feature_encoders[result_feature_names[0]].classes_)],
            dtype=numpy.float64)
        for res_name in result_feature_names:
            vote_count += self.dataset.filter_features([res_name]).encode_target()[0]
        vote_count /= float(len(result_feature_names))

        res_full_ensemble = vote_count.argmax(axis=1)
        instr_res_full_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(
            res_full_ensemble)

        vote_dict = {PROB_CAT_TEMPLATE.format(target=self.target_feature,
                                              clf=self.classifier_name,
                                              cat=cat1,
                                              ): vote_count[:, ix1] for ix1, cat1 in
                     enumerate(self.dataset._feature_encoders['instrument'].classes_)}
        vote_dict.update({'id': self.dataset['id'],
                          MAX_PROB_FEATURE_NAME.format(target=self.target_feature): instr_res_full_ensemble,
                          })
        vote_df = pandas.DataFrame(vote_dict)
        self.dataset.xbt_df = self.dataset.xbt_df.merge(vote_df, on='id')

    def _evaluate_vote_probs(self, xbt_ens_working, xbt_ens_unseen, scores):
        res_ens_working = {'id': xbt_ens_working.xbt_df['id']}
        res_ens_unseen = {'id': xbt_ens_unseen.xbt_df['id']}
        vote_count_working = numpy.zeros(
            [xbt_ens_working.shape[0], self.xbt_labelled._feature_encoders[self.target_feature].classes_.shape[0]],
            dtype=numpy.float64)
        vote_count_unseen = numpy.zeros(
            [xbt_ens_unseen.shape[0], self.xbt_labelled._feature_encoders[self.target_feature].classes_.shape[0]],
            dtype=numpy.float64)
        result_feature_names = []
        # classifications_df = None
        for split_num, estimator in enumerate(scores['estimator']):
            res_name = RESULT_FEATURE_TEMPLATE.format(
                target=self.target_feature,
                clf=self.classifier_name,
                split_num=split_num)
            result_feature_names += [res_name]
            res_ml1_working = estimator.predict(
                xbt_ens_working.filter_features(self.input_features).get_ml_dataset()[0])
            res2_working = xbt_ens_working._feature_encoders[self.target_feature].inverse_transform(
                res_ml1_working).reshape(-1, 1)
            res_ens_working[res_name] = res2_working.reshape(-1)
            vote_count_working += xbt_ens_working._target_encoders[self.target_feature].transform(res2_working)

            res_ml1_unseen = estimator.predict(xbt_ens_unseen.filter_features(self.input_features).get_ml_dataset()[0])
            res2_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].inverse_transform(
                res_ml1_unseen).reshape(-1, 1)
            res_ens_unseen[res_name] = res2_unseen.reshape(-1)
            vote_count_unseen += xbt_ens_unseen._target_encoders[self.target_feature].transform(res2_unseen)

        df_ens_working = pandas.DataFrame(res_ens_working)
        df_ens_unseen = pandas.DataFrame(res_ens_unseen)

        vote_count_working /= float(len(res_ens_working.keys()))
        vote_count_unseen /= float(len(res_ens_working.keys()))

        max_prob_feature_name = MAX_PROB_FEATURE_NAME.format(target={self.target_feature})
        res_working_ensemble = vote_count_working.argmax(axis=1)
        instr_res_working_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(
            res_working_ensemble)
        df_ens_working[max_prob_feature_name] = instr_res_working_ensemble
        res_unseen_ensemble = vote_count_unseen.argmax(axis=1)
        instr_res_unseen_ensemble = self.xbt_labelled._feature_encoders[self.target_feature].inverse_transform(
            res_unseen_ensemble)
        df_ens_unseen[max_prob_feature_name] = instr_res_unseen_ensemble

        df_ens_working = pandas.merge(df_ens_working, xbt_ens_working.xbt_df[['id', 'year']])
        df_ens_unseen = pandas.merge(df_ens_unseen, xbt_ens_unseen.xbt_df[['id', 'year']])

        metric_list_ens = []
        for year in range(self.year_range[0], self.year_range[1]):
            y_ens_working = \
            xbt_ens_working.filter_obs({'year': year}).filter_features([self.target_feature]).get_ml_dataset()[0]
            y_ens_unseen = \
            xbt_ens_unseen.filter_obs({'year': year}).filter_features([self.target_feature]).get_ml_dataset()[0]

            y_res_working = xbt_ens_working._feature_encoders[self.target_feature].transform(
                df_ens_working[df_ens_working.year == year][max_prob_feature_name])
            y_res_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].transform(
                df_ens_unseen[df_ens_unseen.year == year][max_prob_feature_name])
            cats = list(xbt_ens_working._feature_encoders[self.target_feature].classes_)
            (prec_ens_working, recall_ens_working, f1_ens_working,
             support_ens_working) = sklearn.metrics.precision_recall_fscore_support(
                y_ens_working, y_res_working, average='micro', labels=range(0, len(cats)))
            (prec_ens_unseen, recall_ens_unseen, f1_ens_unseen,
             support_ens_unseen) = sklearn.metrics.precision_recall_fscore_support(
                y_ens_unseen, y_res_unseen, average='micro', labels=range(0, len(cats)))
            column_template = '{metric}_{data}_{subset}'
            metric_dict = {'year': year,
                           column_template.format(data='ens_working', metric='precision',
                                                  subset='all'): prec_ens_working,
                           column_template.format(data='ens_working', metric='recall',
                                                  subset='all'): recall_ens_working,
                           column_template.format(data='ens_working', metric='f1', subset='all'): f1_ens_working,
                           column_template.format(data='ens_working', metric='support',
                                                  subset='all'): support_ens_working,
                           column_template.format(data='ens_unseen', metric='precision', subset='all'): prec_ens_unseen,
                           column_template.format(data='ens_unseen', metric='recall', subset='all'): recall_ens_unseen,
                           column_template.format(data='ens_unseen', metric='f1', subset='all'): f1_ens_unseen,
                           column_template.format(data='ens_unseen', metric='support',
                                                  subset='all'): support_ens_unseen,
                           }

            metric_list_ens += [metric_dict]
        metrics_df_ens = pandas.DataFrame.from_records(metric_list_ens)

        # calculate scores for the ensemble classifier output
        cats = list(xbt_ens_working._feature_encoders[self.target_feature].classes_)
        y_ens_working = xbt_ens_working.filter_features([self.target_feature]).get_ml_dataset()[0]
        y_res_working = xbt_ens_working._feature_encoders[self.target_feature].transform(
            df_ens_working[max_prob_feature_name])
        (prec_ens_working, recall_ens_working, f1_ens_working,
         support_ens_working) = sklearn.metrics.precision_recall_fscore_support(y_ens_working, y_res_working,
                                                                                average='micro',
                                                                                labels=range(0, len(cats)))

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
        y_res_unseen = xbt_ens_unseen._feature_encoders[self.target_feature].transform(
            df_ens_unseen[max_prob_feature_name])
        (prec_ens_unseen, recall_ens_unseen, f1_ens_unseen,
         support_ens_unseen) = sklearn.metrics.precision_recall_fscore_support(y_ens_unseen, y_res_unseen,
                                                                               average='micro',
                                                                               labels=range(0, len(cats)))
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
        self.classifiers_export_path_list = []
        for split_num, clf1 in self.classifiers.items():
            export_fname = CLASSIFIER_EXPORT_FNAME_TEMPLATE.format(
                split_num=split_num,
                exp=self.experiment_name,
            )
            self.classifier_output_fnames += [export_fname]
            export_path = os.path.join(self.exp_output_dir,
                                       export_fname)
            joblib.dump(clf1, export_path)
            self.classifiers_export_path_list += [export_path]
        out_dict = dict(self.json_params)
        out_dict['experiment_name'] = out_dict['experiment_name'] + '_inference'
        out_dict['classifier_fnames'] = self.classifier_output_fnames
        self.inference_out_json_path = os.path.join(self.exp_output_dir,
                                                    f'xbt_param_{self.experiment_name}_inference.json')
        print(f' writing inference experiment output file to {self.inference_out_json_path}')
        with open(self.inference_out_json_path, 'w') as json_out_file:
            json.dump(out_dict, json_out_file)


class SingleExperiment(ClassificationExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)

    def _get_classifier(self):
        clf = self.classifier_class(**self._default_classifier_opts)
        return clf

    def run_experiment(self, write_results=True, write_predictions=True,
                       export_classifiers=True):
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

        clf1 = self. _get_classifier()
        clf1.fit(X_dict['train'], y_dict['train'])


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
            self.metrics_out_path = os.path.join(self.exp_output_dir,
                                                 RESULT_FNAME_TEMPLATE.format(name=out_name))
            self.results.to_csv(self.metrics_out_path)
            self.scores_out_path = os.path.join(self.exp_output_dir,
                                                SCORE_FNAME_TEMPLATE.format(name=out_name))
            self.score_table.to_csv(self.scores_out_path)
        else:
            self.metrics_out_path = None
            self.scores_out_path = None

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
            self.predictions_out_path_list = self.dataset.output_data(
                self.exp_output_dir,
                fname_template=OUTPUT_FNAME_TEMPLATE,
                exp_name=out_name,
                output_split=self.output_split,
                target_features=[feature_name],
            )
        else:
            self.predictions_out_path_list = []

        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()

        return (self.results, self.classifiers)

class ImportanceExperiment(ClassificationExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)

    def _get_classifier(self):
        clf = self.classifier_class(**self._default_classifier_opts)
        return clf

    def run_experiment(self, write_results=True, write_predictions=True,
                       export_classifiers=True):
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
        X_dict, y_dict, df_dict, feature_names = self.get_train_test_unseen_sets()

        # fit classifier
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('training classifier')

        clf1 = self. _get_classifier()
        clf1.fit(X_dict['train'], y_dict['train'])

        fi1 = sklearn.inspection.permutation_importance(
            clf1,
            X_dict['test'],
            y_dict['test'],
            n_repeats=IMPORTANCE_REPEATS,
            random_state=0
        )
        importance_df = pandas.DataFrame(
            {'instrument': feature_names,
             'importances_mean': fi1.importances_mean,
             'importances_stdev': fi1.importances_std,
             }
        )

        self.results = importance_df
        self.classifiers = {0: clf1}

        if write_results:
            out_name = self.experiment_name + '_' + self._exp_datestamp
            self.metrics_out_path = os.path.join(self.exp_output_dir,
                                                 IMPORTANCE_FNAME_TEMPLATE.format(
                                                     name=out_name))
            self.results.to_csv(self.metrics_out_path)
        else:
            self.metrics_out_path = None

        self.scores_out_path = None
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        return (self.results, self.classifiers)

class HptExperiment(SingleExperiment):
    def _get_classifier(self):
        classifier_obj = self.classifier_class(**self._default_classifier_opts)
        self.cv_splitter = sklearn.model_selection.KFold(
            n_splits=self.num_training_splits,
            shuffle=True,
            random_state=self._random_state,
        )
        grid_search_cv = sklearn.model_selection.GridSearchCV(
            classifier_obj,
            param_grid=self._tuning_dict['param_grid'],
            scoring=self._tuning_dict['scoring'],
            cv=self.cv_splitter,
        )
        return grid_search_cv

    def run_experiment(self, write_results=True, write_predictions=True,
                       export_classifiers=True):
        super().run_experiment(write_results, write_predictions,
                               export_classifiers)
        if write_results:
            self.best_hp = self.classifiers[0].best_estimator_.get_params()
            # output results to a file
            out_name = self.experiment_name + '_' + self._exp_datestamp

            self.hpt_out_path = os.path.join(self.exp_output_dir,
                                             PARAM_FNAME_TEMPLATE.format(
                                                 name=out_name))
            with open(self.hpt_out_path, 'w') as json_hp_file:
                json.dump(self.best_hp, json_hp_file)




class EnsembleExperiment(ClassificationExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.test_fold_name = 'test_fold'
        self.classifiers = None

    def run_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()

        start1 = time.time()
        print('loading dataset')
        self.load_dataset()
        self.setup_metrics()

        # generate the subset of of the dataset for which we will be able
        # to apply the trained classifiers. The profiles where we can't use
        # the classifier is where a value of a categorical inputs
        # (primarily country), that is present in the unlabelled data is not
        # present in the labelled data, and so the classifier doesn't know
        # what to do with it.
        self.generate_predictable_subset()

        # calculate imeta for the dataset
        self.dataset.xbt_df = pandas.merge(
            self.dataset.xbt_df,
            self.generate_imeta(self.dataset),
            on='id',
        )
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        self.xbt_labelled.xbt_df = pandas.merge(
            self.xbt_labelled.xbt_df,
            self.generate_imeta(self.xbt_labelled),
            on='id',
        )
        self.xbt_labelled._feature_encoders.update(
            {f'imeta_{self.target_feature}': self.xbt_labelled._feature_encoders[
                self.target_feature]})

        # get train/test/unseen sets
        print('generating splits')

        self.generate_test_working_datasets()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        self.train_classifiers()

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating probabilities for evaluation.')

        # generate ensemble outputs for calculating metrics on test set
        ensemble_output_test = self.generate_ensemble_predictions(self.xbt_test)

        self.xbt_test.xbt_df = pandas.merge(
            self.xbt_test.xbt_df,
            ensemble_output_test,
            on='id',
        )

        self.xbt_test._feature_encoders.update(
            {col_name: self.xbt_test._feature_encoders[self.target_feature]
            for col_name in ensemble_output_test.columns if 'res' in col_name}
        )

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating global metrics.')
        self.calculate_global_metrics(self.xbt_test)

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating per year metrics.')
        self. calculate_annual_metrics(self.xbt_test)

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating per class metrics.')
        self.calculate_per_class_metrics(self.xbt_test)

        pred_df = self.generate_ensemble_predictions(self.xbt_predictable)
        self.output_quality_flag_name = OUTPUT_CQ_FLAG.format(
            var_name=self.target_feature)
        self.best_data_feature_name = f'res_{self.target_feature}_best_data'
        pred_df = pandas.merge(
            pred_df,
            pandas.DataFrame({
                'id': self.xbt_labelled['id'],
                self.output_quality_flag_name: [OUTPUT_CQ_INPUT] * self.xbt_labelled.shape[0],
                self.best_data_feature_name: self.xbt_labelled[self.target_feature],
             }),
            on='id',
            how='outer')
        ml_output_pred_indices = pred_df[pred_df[self.output_quality_flag_name].isna()].index
        pred_df.loc[ml_output_pred_indices, self.output_quality_flag_name] = OUTPUT_CQ_ML
        pred_df.loc[ ml_output_pred_indices, self.best_data_feature_name,] = pred_df.loc[ml_output_pred_indices,self.max_prob_feature_name]

        probs_column_names = [c1 for c1 in pred_df.columns
                              if 'probability' in c1]

        results_column_names = [c1 for c1 in pred_df.columns
                                if 'res' in c1]
        missing_data_indices = self.dataset.xbt_df[self.dataset.xbt_df.is_predictable == 0].index

        self.dataset.xbt_df = pandas.merge(
            self.dataset.xbt_df,
            pred_df,
            on='id',
            how='outer')

        #use 0.0 for missing probabilities
        for col1 in probs_column_names:
            self.dataset.xbt_df.loc[missing_data_indices, col1] = 0.0

        # fill in imeta for missing data
        for col1 in results_column_names:
            self.dataset.xbt_df.loc[missing_data_indices, col1] = self.dataset.xbt_df.loc[missing_data_indices, self.imeta_feature_name]

        if write_results:
            for split_name, split_results in self.results.items():
                for cat_name, cat_results in split_results.items():
                    out_name = f'{self.experiment_name}_{split_name}_' \
                               f'{cat_name}_{self._exp_datestamp}'
                    metrics_out_path = os.path.join(
                        self.exp_output_dir,
                        RESULT_FNAME_TEMPLATE.format(
                            name=out_name))
                    cat_results.to_csv(metrics_out_path)

        if export_classifiers:
            print('exporting classifier objects through pickle')
            self.export_classifiers()

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('output predictions')
        if write_predictions:
            out_name = self.experiment_name + '_cv_' + self._exp_datestamp
            self.predictions_out_path_list = self.dataset.output_data(
                self.exp_output_dir,
                fname_template=OUTPUT_FNAME_TEMPLATE,
                exp_name=out_name,
                output_split=self.output_split,
                target_features=[],
            )
        else:
            self.predictions_out_path_list = []


        return (self.results, self.classifiers)

    def train_classifiers(self):
        raise NotImplementedError('train_classifier - This method of the '
                                  'abstract base class should be overloaded '
                                  'by concrete class implementation.')

    def generate_ensemble_predictions(self, xbt_subset):
        d1 = {
            f'{self.target_feature}_res_{ix1}':
                xbt_subset._feature_encoders[
                    self.target_feature].inverse_transform(
                    clf1.predict(xbt_subset.filter_features(
                        self.input_features).get_ml_dataset()[0]))
            for ix1, clf1 in self.classifiers.items()}
        d1['id'] = xbt_subset['id']

        ens_res_features = [f'{self.target_feature}_res_{ix1}' for
                               ix1, clf1 in self.classifiers.items()]

        probs_array = functools.reduce(
            lambda x, y: x + y,
            [self.xbt_labelled._target_encoders[self.target_feature].transform(
                pred1.reshape(-1,1))
             for col_name, pred1 in d1.items()
             if 'res' in col_name]) / len(self.classifiers)

        probs_dict = {col_name : probs_array[:, col_ix1] for col_ix1, col_name
                      in enumerate(self.xbt_labelled._target_encoders[
                                       self.target_feature].categories_[0])}
        d1.update(
            {f'probability_{self.target_feature}_{k1}':v1 for k1,v1 in probs_dict.items()}
        )
        probs_df = pandas.DataFrame(probs_dict)

        max_prob_ensemble = probs_df.idxmax(axis='columns')
        self.max_prob_feature_name = MAX_PROB_FEATURE_NAME.format(target=self.target_feature)
        d1[self.max_prob_feature_name] = max_prob_ensemble

        ensemble_outputs = pandas.DataFrame({k1: list(v1) for k1,v1 in d1.items()})
        return ensemble_outputs

    def calculate_global_metrics(self, xbt_subset):
        metrics_test_ens = pandas.DataFrame(
            {f'{metric_name}_instr_ens': [
                self.predict_and_calc_metric_subset(xbt_subset,
                                        clf1,
                                        None,
                                        **metric1
                                        ) for res_ix1, clf1 in
                self.classifiers.items()] + [
                self.calc_column_metric_subset(xbt_subset,
                                               self.max_prob_feature_name,
                                               None,
                                               **metric1
                                               )
            ] + [
                self.calc_column_metric_subset(xbt_subset,
                                               f'imeta_{self.target_feature}',
                                               None,
                                               **metric1
                                               )
            ]

                for metric_name, metric1 in self.metrics_defs_dict.items()
            })
        metrics_test_ens['classifier'] = [f'clf_{res_ix1}' for res_ix1, clf1 in self.classifiers.items()] + ['max_prob', 'imeta']
        self.results[TEST_VAR_NAME][METRIC_SET_ALL] = metrics_test_ens

    def calculate_per_class_metrics(self, xbt_subset):
        metrics_raw_dict = {
            f'{metric_name}_instr_{res_ix1}': [
                self.predict_and_calc_metric_subset(xbt_subset,
                                        clf1,
                                        {self.target_feature: fn1},
                                        **metric1
                                        ) for fn1
                                               in self.instrument_list]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            for res_ix1, clf1 in self.classifiers.items()
            }
        metrics_raw_dict.update({
            f'{metric_name}_instr_max_prob': [
                self.calc_column_metric_subset(xbt_subset,
                                               self.max_prob_feature_name,
                                               {self.target_feature: fn1},
                                               **metric1,
                                               )
                for fn1 in self.instrument_list]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            })
        metrics_raw_dict.update({
            f'{metric_name}_instr_imeta': [
                self.calc_column_metric_subset(xbt_subset,
                                               f'imeta_{self.target_feature}',
                                               {self.target_feature: fn1},
                                               **metric1,
                                               )
                for fn1 in self.instrument_list]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            })


        metrics_raw_dict['num_profiles'] = [
            self.xbt_labelled.filter_obs({self.target_feature: fn1}).shape[0]
            for fn1 in self.instrument_list]
        metrics_raw_dict[self.target_feature] = [fn1 for fn1 in
                                                    self.instrument_list]

        metrics_per_class_df = pandas.DataFrame(metrics_raw_dict)
        metrics_per_class_df = metrics_per_class_df.sort_values(
            'num_profiles', ascending=False)
        for metric_name in self.metrics_defs_dict.keys():
            metrics_per_class_df[f'{metric_name}_instr_avg'] = \
            metrics_per_class_df[
                [c1 for c1 in metrics_per_class_df.columns if
                 metric_name in c1]].mean(axis='columns')
        self.results[TEST_VAR_NAME][METRIC_SET_PER_CLASS] = metrics_per_class_df

    def calculate_annual_metrics(self, xbt_subset):
        metrics_annual_raw_dict = {
            f'{metric_name}_instr_{res_ix1}': [self.predict_and_calc_metric_subset(xbt_subset,
                                                                  clf1,
                                                                  {'year': year1},
                                                                  **metric1
                                                                  )
                                               for year1 in range(*self.year_range)]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            for res_ix1, clf1 in self.classifiers.items()
            }
        max_prob_metrics = {
            f'{metric_name}_instr_max_prob': [
                self.calc_column_metric_subset(xbt_subset,
                                               self.max_prob_feature_name,
                                               {'year': year1},
                                               **metric1,
                                               )
                for year1 in range(*self.year_range)]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            }
        metrics_annual_raw_dict.update(max_prob_metrics)

        [self.calc_column_metric_subset(xbt_subset, self.max_prob_feature_name, {'year': year1}, **self.metrics_defs_dict['recall'],) for year1 in range(*self.year_range)]

        metrics_annual_raw_dict.update({
            f'{metric_name}_instr_imeta': [
                self.calc_column_metric_subset(xbt_subset,
                                               f'imeta_{self.target_feature}',
                                               {'year': year1},
                                               **metric1,
                                               )
                for year1 in range(*self.year_range)]
            for metric_name, metric1 in self.metrics_defs_dict.items()
            })
        metrics_annual_raw_dict['num_profiles'] = [
            xbt_subset.filter_obs({'year': year1}).shape[0] for year1 in
            range(*self.year_range)]
        metrics_annual_raw_dict['year'] = [year1 for year1 in
                                           range(*self.year_range)]

        metrics_annual_df = pandas.DataFrame(metrics_annual_raw_dict)
        for metric_name in self.metrics_defs_dict.keys():
            metrics_annual_df[f'{metric_name}_instr_avg'] = \
            metrics_annual_df[
                [c1 for c1 in metrics_annual_df.columns if
                 metric_name in c1]].mean(axis='columns')
        self.results[TEST_VAR_NAME][METRIC_SET_PER_YEAR] = metrics_annual_df


class CVExperiment(EnsembleExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)

    def train_classifiers(self):
        self.xbt_working.generate_folds_by_feature('cruise_number',
                                                   self.num_test_splits,
                                                   self.test_fold_name)
        cruise_numbers = list(self.xbt_working['cruise_number'].unique())

        X_labelled = \
            self.xbt_working.filter_features(self.input_features).get_ml_dataset()[
                0]
        y_labelled = self.xbt_working.filter_features(
            [self.target_feature]).get_ml_dataset()[0]

        group_cv1 = sklearn.model_selection.KFold(
            n_splits=self.num_test_splits,
            shuffle=True,
            random_state=random.randint(
                1, 2 ** 20))
        clf_dt1 = self.classifier_class(**self._default_classifier_opts)

        scores = sklearn.model_selection.cross_validate(
            clf_dt1,
            X_labelled, y_labelled,
            groups=self.xbt_working[self.test_fold_name],
            cv=group_cv1,
            return_estimator=True,
            return_train_score=True,
            scoring=self._tuning_dict['cv_metrics'],
            n_jobs=-1,
        )

        self.classifiers = {ix1: clf1
                            for ix1, clf1 in enumerate(scores['estimator'])}


class CvhptExperiment(EnsembleExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)

    def train_classifiers(self):
        self.xbt_working.generate_folds_by_feature('cruise_number',
                                                   self.num_test_splits,
                                                   self.test_fold_name)
        cruise_numbers = list(self.xbt_working['cruise_number'].unique())

        X_labelled = \
            self.xbt_working.filter_features(self.input_features).get_ml_dataset()[
                0]
        y_labelled = self.xbt_working.filter_features(
            [self.target_feature]).get_ml_dataset()[0]

        self.test_splitter = sklearn.model_selection.KFold(
            n_splits=self.num_test_splits,
            shuffle=True,
            random_state=random.randint(
                1, 2 ** 20))

        self.train_splitter = sklearn.model_selection.KFold(
            n_splits=self.num_training_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        classifier_obj = self.classifier_class(**self._default_classifier_opts)
        grid_search_cv = sklearn.model_selection.GridSearchCV(
            classifier_obj,
            param_grid=self._tuning_dict['param_grid'],
            scoring=self._tuning_dict['scoring'],
            cv=self.train_splitter,
        )

        scores = sklearn.model_selection.cross_validate(
            grid_search_cv,
            X_labelled, y_labelled,
            groups=self.xbt_working[self.test_fold_name],
            cv=self.test_splitter,
            return_estimator=True,
            return_train_score=True,
            scoring=self._tuning_dict['cv_metrics'],
            n_jobs=-1,
        )

        self.classifiers = {ix1: clf1
                            for ix1, clf1 in enumerate(scores['estimator'])}

class ResamplingExperiment(EnsembleExperiment):
    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.num_resamples_per_class = 10000

    def train_classifiers(self):
        self.num_resamples_per_class = int(self.xbt_labelled.shape[0] / len(self.instrument_list))


        self.classifiers = {}
        for resample_index in range(1, 6):
            xbt_resampled_train_all, xbt_resampled_validation_all = self.get_resampled(
                self.xbt_working,
                resample_index=resample_index,
                random_state=int(
                    (datetime.datetime.now().timestamp() * 1e5) % 1e5),
            )
            clf1 = self.classifier_class(**self._default_classifier_opts)
            clf1.fit(xbt_resampled_train_all.filter_features(
                self.input_features).get_ml_dataset()[0],
                     xbt_resampled_train_all.filter_features(
                         [self.target_feature]).get_ml_dataset()[0],
                     )
            self.classifiers[resample_index] = clf1

    def get_resampled(self, xbt_subset1, resample_index, random_state):
        resampled_profiles_list = [
            xbt_subset1.filter_obs({self.target_feature: ins1}).xbt_df.sample(
                self.num_resamples_per_class,
                replace=True,
                random_state=random_state,
            )
            for ins1 in self.instrument_list]
        resampled_training_indices = list(set(itertools.chain.from_iterable([list(rp1.index) for rp1 in resampled_profiles_list])))
        if resample_index is None:
            resample_index = 0
        resample_feature_name = RESAMPLE_FEATURE_TEMPLATE.format(resample_index=resample_index)
        xbt_subset1.xbt_df[resample_feature_name] = self.xbt_working.xbt_df.index.isin(resampled_training_indices)
        resampled_df = pandas.concat(
            resampled_profiles_list,
            ignore_index=True,
        )
        xbt_resampled_train_all = xbt.dataset.XbtDataset(self.data_dir, self.year_range, df=resampled_df)
        xbt_resampled_train_all._feature_encoders = self.xbt_labelled._feature_encoders
        xbt_resampled_train_all._arget_encoders = self.xbt_labelled._target_encoders
        xbt_resampled_validation_all = self.xbt_working.filter_obs({resample_feature_name: False})
        return (xbt_resampled_train_all, xbt_resampled_validation_all)


class InferenceExperiment(ClassificationExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)

    def run_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
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

