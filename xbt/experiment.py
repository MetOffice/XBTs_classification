import ast
import re
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
CLASSIFIER_EXPORT_FNAME_REGEX = 'xbt_classifier_(?P<clf_name>[a-zA-Z0-9_]+).joblib'

CLF_KEY_TEMPLATE = 'clf_{ens_type}_{index}'
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

    def generate_predictable_subset(self):
        if self.xbt_predictable is None:
            checkers_labelled = {f1: c1 for f1, c1 in
                                 self.xbt_labelled.get_checkers().items() if
                                 f1 in self.input_features}

            self.xbt_predictable = self.dataset.filter_predictable(
                checkers_labelled)
            self.xbt_predictable._feature_encoders = self.xbt_labelled._feature_encoders
            self.xbt_predictable._target_encoders = self.xbt_labelled._target_encoders



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


class EnsembleExperiment(ClassificationExperiment):
    """
    """
    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.test_fold_name = 'test_fold'
        self.classifiers = None

    def run_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        print(f'Running {self.exp_type} on config {self.json_fname}')
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()

        start1 = time.time()

        self.setup_experiment()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        self.train_classifiers()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating probabilities for evaluation.')

        self.calculate_metrics(start1)
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        self.do_predictions()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

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

    def setup_experiment(self):
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


    def train_classifiers(self):
        raise NotImplementedError('train_classifier - This method of the '
                                  'abstract base class should be overloaded '
                                  'by concrete class implementation.')

    def calculate_metrics(self, start1=None):
        if start1 is None:
            start1 = time.time()
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

    def do_predictions(self):
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
        self.exp_type = 'Cross validation experiment'

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

        self.classifiers = {CLF_KEY_TEMPLATE.format(ens_type='cv', index=ix1): clf1
                            for ix1, clf1 in enumerate(scores['estimator'])}


class CvhptExperiment(EnsembleExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.exp_type = 'Cross-validation experiment with hyperparameter tuning'

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
        self.exp_type = 'Resampling experiment'

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
            self.classifiers[CLF_KEY_TEMPLATE.format(ens_type='resample', index=resample_index)] = clf1

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

class MetaEnsembleExperiment(EnsembleExperiment):
    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.num_resamples_per_class = 10000
        self.resample_exp = ResamplingExperiment(json_descriptor=json_descriptor,
                                                 data_dir=data_dir,
                                                 output_dir=output_dir,
                                                 output_split=output_split,
                                                 do_preproc_extract=do_preproc_extract,
                                                 )

        self.kfold_exp = CVExperiment(json_descriptor=json_descriptor,
                                      data_dir=data_dir,
                                      output_dir=output_dir,
                                      output_split=output_split,
                                      do_preproc_extract=do_preproc_extract,
                                      )
        self.exp_type = 'Meta-ensemble experiment'


    def train_classifiers(self):
        self.resample_exp.xbt_labelled = self.xbt_labelled
        self.resample_exp.xbt_working = self.xbt_working
        self.resample_exp.instrument_list = self.instrument_list
        self.resample_exp.train_classifiers()

        self.kfold_exp.xbt_working = self.xbt_working
        self.kfold_exp.xbt_labelled = self.xbt_labelled
        self.kfold_exp.instrument_list = self.instrument_list
        self.kfold_exp.train_classifiers()

        self.classifiers = {}
        self.classifiers.update(self.kfold_exp.classifiers)
        self.classifiers.update(self.resample_exp.classifiers)





class InferenceExperiment(EnsembleExperiment):

    def __init__(self, json_descriptor, data_dir, output_dir, output_split, do_preproc_extract=False):
        super().__init__(json_descriptor, data_dir, output_dir, output_split, do_preproc_extract)
        self.exp_type = 'Inference experiment'

    def run_experiment(self, write_results=True, write_predictions=True, export_classifiers=True):
        """
        """
        print(f'Running {self.exp_type} on config {self.json_fname}')
        self._check_output_dir()
        self._exp_datestamp = xbt.common.generate_datestamp()
        start1 = time.time()

        self.setup_experiment()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('loading classifiers from files.')

        self.train_classifiers()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('generating predictions and probabilities.')

        self.do_predictions()
        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')

        duration1 = time.time() - start1
        print(f'{duration1:.3f} seconds since start.')
        print('output predictions')
        print('writing predictions and probabilities to file.')
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

        return (None, self.classifiers)


    def train_classifiers(self):
        print('loading saved classifiers from pickle files.')
        pattern1 = 'xbt_classifier_(?P<name>[a-zA-Z0-9_]+).joblib'
        self.classifiers = {
            re.search(CLASSIFIER_EXPORT_FNAME_REGEX, fname1).group('clf_name') :
                joblib.load(os.path.join(self.experiment_description_dir,
                                          fname1))
            for ix1, fname1 in enumerate(self.classifier_fnames)
            }

