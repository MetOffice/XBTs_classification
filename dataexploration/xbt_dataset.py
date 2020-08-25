import os
import re
import pandas
import functools
import datetime
import dask.dataframe
import numpy
import tempfile
import sklearn.preprocessing

import preprocessing.extract_year
import xbt.common
XBT_FNAME_TEMPLATE = 'xbt_{year}.csv'
XBT_CSV_REGEX_STR = 'xbt_(?P<year>[0-9]+).csv'

INSTRUMENT_REGEX_STRING = 'XBT[:][\s](?P<model>[\w\s;:-]+)([\s]*)([(](?P<manufacturer>[\w\s.:;-]+)[)])?'
REGEX_MANUFACTURER_GROUP = 'manufacturer'
REGEX_MODEL_GROUP = 'model'

UNKNOWN_STR = 'UNKNOWN'
UNKNOWN_MODEL_STR = 'TYPE UNKNOWN'
UNKNOWN_MANUFACTURER_STR = 'UNKNOWN BRAND'

PREDICTABLE_FLAG = 'is_predictable'

EXCLUDE_LIST = ['Unnamed: 0']
KEY_DICT = {
    'CRUISE': 'cruise_number',
}

def get_model(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        type_str = matches.group(REGEX_MODEL_GROUP)
    except AttributeError as e1:
        type_str = UNKNOWN_STR
    return str(type_str).strip(' ')


def get_manufacturer(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        brand_str = matches.group(REGEX_MANUFACTURER_GROUP)
    except AttributeError as e1:
        brand_str = UNKNOWN_STR
    return str(brand_str).strip(' ')

def check_value_found(ref, value):
    if ref in value:
        return True
    return False

def check_cat_value_allowed(allowed, value):
    return value in allowed

def normalise_lat(feature_lat, do_transform=True):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.data_min_ = -90.0
    encoder.data_max_ = 90.0
    if do_transform:
        ml_feature = encoder.transform(feature_lat)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

def normalise_lon(feature_lon, do_transform=True):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.data_min_ = -180.0
    encoder.data_max_ = 180.0
    if do_transform:
        ml_feature = encoder.transform(feature_lon)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

def get_cat_ml_feature(cat_feature, do_transform=True):
    encoder = sklearn.preprocessing.OneHotEncoder(
        sparse=False,
        handle_unknown='ignore' #we set this to ignore, so when applying the encoder we don't get an exception for missing values.
    )
    encoder.fit(cat_feature)
    if do_transform:
        ml_feature = encoder.transform(cat_feature)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

class XbtLabelEncoder(sklearn.preprocessing.LabelEncoder):
    def transform(self, ferature_array):
        return super(XbtLabelEncoder, self).transform(ferature_array).reshape(-1,1)

def get_target_ml_feature(target_feature, do_transform=True):
    encoder = XbtLabelEncoder()
    encoder.fit(target_feature)
    if do_transform:
        ml_feature = encoder.transform(target_feature)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

def get_ord_ml_feature(ord_feature, do_transform=True):
    encoder = sklearn.preprocessing.OrdinalEncoder()
    encoder.fit(ord_feature)
    if do_transform:
        ml_feature = encoder.transform(ord_feature)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

def get_num_ml_feature(numerical_feature, do_transform=True):
    encoder = sklearn.preprocessing.StandardScaler()
    encoder.fit(numerical_feature)
    if do_transform:
        ml_feature = encoder.transform(numerical_feature)
    else:
        ml_feature = None    
    return (encoder, ml_feature)

def get_minmaxfixed_ml_feature(data_min, data_max, minmax_feature, do_transform=True):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.fit(minmax_feature)
    encoder.data_min_ = data_min
    encoder.data_max_ = data_max
    if do_transform:
        ml_feature = encoder.transform(minmax_feature)
    else:
        ml_feature = None
    return (encoder, ml_feature)

def get_minmax_ml_feature(minmax_feature, do_transform=True):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.fit(minmax_feature)
    if do_transform:
        ml_feature = encoder.transform(minmax_feature)
    else:
        ml_feature = None
    return (encoder, ml_feature)

def cat_output_formatter(feature_name, feature_data, encoder):
    feat_arr = encoder.transform(feature_data)
    out_columns = [f'{feature_name}_{col_name}' for col_name in encoder.categories_[0]]
    return pandas.DataFrame(feat_arr, columns=out_columns).astype(
        dtype={f1: 'int8' for f1 in out_columns})

CATEGORICAL_LOADED_FEATURES = [
    'country', 'institute', 'platform', 'cruise_number', 'imeta_applied',
]
CATEGORICAL_GENERATED_FEATURES = []
CATEGORICAL_FEATURES = CATEGORICAL_LOADED_FEATURES + CATEGORICAL_GENERATED_FEATURES

TARGET_LOADED_FEATURES = ['instrument', 'model', 'manufacturer', ]
TARGET_GENERATED_FEATURES = []
TARGET_FEATURES = TARGET_LOADED_FEATURES + TARGET_GENERATED_FEATURES

PROFILE_FEATURES = ['temperature_profile', 'depth_profile']
QUALITY_FLAG_FEATURES = ['temperature_quality_flag', 'depth_quality_flag']
ID_FEATURES = ['id']
ORDINAL_FEATURES = ['year', 'month', 'day']
MINMAX_FEATURES = ['max_depth']
NORMAL_DIST_FEATURES = []
CUSTOM_FEATURES = ['lat', 'lon']
OTHER_FEATURES = ['date']
XBT_CONVERTERS = {'temperature_profile': eval,
                  'depth_profile': eval,
                 }


FEATURE_PROCESSORS = {}
FEATURE_PROCESSORS.update({f1: get_cat_ml_feature for f1 in CATEGORICAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_target_ml_feature for f1 in TARGET_FEATURES})
FEATURE_PROCESSORS.update({f1: get_ord_ml_feature for f1 in ORDINAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_minmax_ml_feature for f1 in MINMAX_FEATURES})
FEATURE_PROCESSORS.update({f1: get_num_ml_feature for f1 in NORMAL_DIST_FEATURES})
FEATURE_PROCESSORS.update({'lat': functools.partial(get_minmaxfixed_ml_feature, -90.0, 90.0),
                           'lon': functools.partial(get_minmaxfixed_ml_feature, -180.0, 180.0),
                           'max_depth': functools.partial(get_minmaxfixed_ml_feature, 0.0, 2000.0),
                          })

TARGET_PROCESSORS = {}
TARGET_PROCESSORS.update({f1: get_cat_ml_feature for f1 in TARGET_FEATURES})


OUTPUT_FORMATTERS = {}
OUTPUT_FORMATTERS.update({f1: [cat_output_formatter] for f1 in CATEGORICAL_FEATURES + TARGET_LOADED_FEATURES})



TRAIN_SET_FEATURE = 'training_set'
ML_EXCLUDE_LIST = ['date', 'id', 'month', 'day', TRAIN_SET_FEATURE]

def read_csv(fname, features_to_load, converters=None):
    return pandas.read_csv(fname,
                           converters=converters,
                           usecols=features_to_load
                          )

def do_concat(df_list, axis=1, ignore_index=True):
    xbt_df = pandas.concat(df_list, ignore_index=ignore_index)
    for feature1 in (CATEGORICAL_FEATURES + TARGET_LOADED_FEATURES):
        xbt_df[feature1] = xbt_df[feature1].astype('category') 
    for feature1 in ID_FEATURES:
        xbt_df[feature1] = xbt_df[feature1].astype('int') 
    return xbt_df


def do_preprocessing(xbt_df):
    # exclude bad dates
    xbt_df = xbt_df[xbt_df['year'] != 0]
    # exclude bad depths
    xbt_df = xbt_df[xbt_df['max_depth'] < 2000.0]
    xbt_df = xbt_df[xbt_df['max_depth'] > 0.0]

    return xbt_df


class XbtDataset():
    def __init__(self, directory, year_range, df=None, use_dask=False, load_profiles=False, load_quality_flags=False, nc_dir=None, pp_prefix='', pp_suffix=''):
        self._use_dask = use_dask
        self._load_profiles = load_profiles
        self._load_quality_flags = load_quality_flags
        self.features_to_load = CATEGORICAL_LOADED_FEATURES + MINMAX_FEATURES + CUSTOM_FEATURES + ID_FEATURES + OTHER_FEATURES + TARGET_LOADED_FEATURES + ORDINAL_FEATURES
        if self._load_profiles:
            self.features_to_load += PROFILE_FEATURES
        if self._load_quality_flags:
            self.features_to_load += QUALITY_FLAG_FEATURES
        if self._use_dask:
            self._read_func = dask.delayed(read_csv)
            self._preproc_func = dask.delayed(do_preprocessing)
            self._concat_func = dask.delayed(do_concat)
        else:
            self._read_func = read_csv
            self._preproc_func = do_preprocessing
            self._concat_func = do_concat
        self.nc_directory = nc_dir # if this is defined then do a preprocessing step
        self.pp_prefix = pp_prefix
        self.pp_suffix = pp_suffix
        
        self.directory = directory
        self.year_range = year_range
        
        #this will be created in load files, which may happen after preprocessing
        self.dataset_files = []
        self.xbt_df = df
        self._feature_encoders = {}
        self._target_encoders = {}
        self._output_formatters = OUTPUT_FORMATTERS
        
        if self.xbt_df is None:
            self._load_data() 

    def _load_data(self):
        if self.year_range is None:
            start_year = None
            end_year = None
        else:
            start_year = self.year_range[0]
            end_year = self.year_range[1]
        
        if self.nc_directory is not None:
            #create a temp subdirectory to be used in the preprocessing, which will then be deleted after preprocessing.
            with tempfile.TemporaryDirectory(dir=self.directory) as temp_dir:
                preprocessing.extract_year.do_wod_extract(
                    nc_dir=self.nc_directory, 
                    out_dir=self.directory, 
                    temp_dir=temp_dir,
                    start_year=start_year, 
                    end_year=end_year, 
                    fname_prefix=self.pp_prefix, 
                    fname_suffix=self.pp_suffix, 
                    pool_size=preprocessing.extract_year.DEFAULT_PREPROC_TASKS,                
            )
            
        
        if self.year_range is None:
            year_list = [int(re.search(XBT_CSV_REGEX_STR, fname1).group('year')) for fname1 in os.listdir(self.directory)]
            start_year = min(year_list)
            end_year = max(year_list)
            self.year_range = (start_year, end_year)
            print(f'derived year range from data: {start_year} to {end_year}')

        self.dataset_files = [os.path.join(self.directory, XBT_FNAME_TEMPLATE.format(year=year)) for year in range(start_year, end_year+1)]
        self.dataset_files = [f1 for f1 in self.dataset_files if os.path.isfile(f1)]
        df_in_list = [self._read_func(year_csv_path, self.features_to_load) for year_csv_path in self.dataset_files]
        df_processed = [self._preproc_func(df_in) for df_in in df_in_list]
        self.xbt_df = self._concat_func(df_processed)
    
    def _get_subset_df(self, filters, mode='include', check_type='match_subset'):
        included_in_subset = True
        xbt_df = self.xbt_df
        filter_outputs = []
        for key, value in filters.items():
            if key == 'labelled':
                if value == 'labelled':
                    check1 = ((xbt_df['imeta_applied'] == 0) & 
                                              (~(xbt_df['instrument'].apply(functools.partial(check_value_found,'UNKNOWN') ).astype(bool))) )
                        
                        
                elif value == 'unlabelled':
                    check1 = ((xbt_df['imeta_applied'] == 1) |  
                                          xbt_df['instrument'].apply(functools.partial(check_value_found,'UNKNOWN') ).astype(bool))
                                         
                elif value == 'imeta':
                    check1 =  (xbt_df['imeta_applied'] == 1)
                elif value == 'all':
                    check1 = (xbt_df['imeta_applied'] == 0) | (xbt_df['imeta_applied'] != 0)
            else:
                if check_type == 'match_subset':
                    try:
                        check1 = xbt_df[key].apply(lambda x: value in x)
                    except TypeError:
                        check1 = (value == xbt_df[key])
                elif check_type == 'in_filter_set':
                    check1 = xbt_df[key].apply(lambda x: x in value)
                              
            filter_outputs += [check1.astype(bool)]
                              
            included_in_subset = functools.reduce(lambda x,y: x & y, filter_outputs)
            included_in_subset = included_in_subset.astype(bool)
               
            if mode == 'exclude':
                included_in_subset = ~included_in_subset 
        
        subset_df = self.xbt_df[included_in_subset] 
        return subset_df
    
    def sample_feature_values(self, feature, fraction, split_feature=None):
        """
        Get a sample of the unique values in a categorical feature. For example,
        if the feature is cruise_number, the fraction is 0.1 and there are 100 
        unique values in the feature cruise_number, you will get a list
        with 10 values of cruise_number.
        """
        if split_feature is None:
            df_values = pandas.DataFrame(self.xbt_df[feature].unique(), columns=[feature])
            sample_values = list(df_values.sample(frac=fraction)[feature])
        else:
            sample_values = []
            for sfv1 in self.xbt_df[split_feature].unique():
                df_values = pandas.DataFrame(self.xbt_df[self.xbt_df[split_feature] == sfv1][feature].unique(), 
                                             columns=[feature])
                sample_values += list(df_values.sample(frac=fraction)[feature])
        return sample_values
        
    def filter_obs(self, filters, mode='include', check_type='match_subset'):
        """
        Filter the observation in the XBT dataset by returning only the rows 
        (observations/profiles) where some element of metadata matches a given 
        value. A common use is to filter by year, in which all profiles from a 
        given year will be in the return XbtDataset object. In addition to the 
        features present in the dataset, one can also use "labelled" as a key. 
        In that case, the value parameter can be "labelled", "unlabelled" or 
        "imeta". For labelled, all labelled data is return. This is data for 
        which the iMeta algorithm has not been applied and which does not 
        have the word "UNKNOWN" in the instrument value. Unlabelled data is 
        all data which has either had the iMeta algorithm applied or which 
        has the word "UNKNOWN" in the instrument value. "imeta" return all 
        profiles which have had the imeta algorithm applied.
        
        values of mode: include, exclude
        values of check_type: match_subset, in_filter_set
        """
        subset_df = self._get_subset_df(filters, mode, check_type)
        filtered_dataset = XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
        filtered_dataset._feature_encoders = self._feature_encoders                
        filtered_dataset._target_encoders = self._target_encoders                
        return filtered_dataset

    def filter_predictable(self, checkers):
        """
        In some cases, once we have trained a classifier, we can't apply it to the whole dataset because values in 
        input features are not present in the labelled data, so the classifier cannot use these values. So this
        function will return a subset of the data that can be used with a classifier that uses the features listed
        in filter features as input.        
        """
        filter_values = self.xbt_df.apply(
            lambda row1: numpy.all([checker_func1(row1[feat1]) for feat1, checker_func1 in checkers.items()]),
            axis='columns',
        )
                
        self.xbt_df[PREDICTABLE_FLAG] = 0
        self.xbt_df.loc[self.xbt_df.index[filter_values], PREDICTABLE_FLAG] = 1
        subset = self.xbt_df[filter_values]
        xbt_predictable = XbtDataset(
            year_range=self.year_range, 
            directory=self.directory, 
            df = subset
            )        
        return xbt_predictable

    @property
    def feature_encoders(self):
        return self._feature_encoders
    
    @property
    def target_encoders(self):
        return self._target_encoders
    
    def _get_features_to_process(self):
        features_to_process = [feature1 for feature1 in self.xbt_df.columns 
            if feature1 not in ML_EXCLUDE_LIST]
        return features_to_process

    def get_checkers(self):
        # first ensure we have encoders
        _ = self.get_ml_dataset(return_data=False)
        checkers = {}
        for f1 in CATEGORICAL_FEATURES:
            checkers[f1] = functools.partial(check_cat_value_allowed, 
                                              list(self.xbt_df[f1].unique()))
        return checkers
    
    def output_data(self, out_dir, fname_template, exp_name, target_features=[], output_split=xbt.common.OUTPUT_SINGLE):
        out_df = self.xbt_df
        for feat1 in target_features:
            try:
                for formatter1 in self._output_formatters[feat1]:
                    out_df = out_df.join(formatter1(feat1, 
                                                    out_df[[feat1]], 
                                                    self._target_encoders[feat1]))
            except KeyError:
                print(f'cannot output ML feature {feat1}, no formatter available.')
        if output_split == xbt.common.OUTPUT_SINGLE:
                        

            out_path = os.path.join(out_dir,
                                    fname_template.format(exp_name=exp_name,
                                                          subset='all')
                                   )
            print(f'output all predictions to {out_path}')
            out_df.to_csv(out_path)
        elif output_split == xbt.common.OUTPUT_YEARLY:
            print('output predictions by year to {0}'.format(
                os.path.join(out_dir, fname_template.format(exp_name=exp_name,
                                                            subset='YYYY')
                                       )))
            for current_year in out_df.year.unique():
                out_path = os.path.join(out_dir,
                                        fname_template.format(exp_name=exp_name,
                                                              subset=f'{current_year:04d}')
                                       )
                out_df[out_df.year == current_year].to_csv(out_path)
        elif output_split == xbt.common.OUTPUT_MONTHLY:
            print('output predictions by month to {0}'.format(
                os.path.join(out_dir, fname_template.format(exp_name=exp_name,
                                                            subset='YYYYMM')
                                       )))
            for current_year in out_df.year.unique():
                for current_month in range(0,12):
                    out_path = os.path.join(out_dir,
                                            fname_template.format(exp_name=exp_name,
                                                                  subset=f'{current_year:04d}{current_month:02d}'))
                    out_df[(out_df.year == current_year) & (out_df.month == current_month)].to_csv(out_path)
    
    def merge_features(self, other, features_to_merge, fill_values=None, feature_encoders=None, target_encoders=None, output_formatters=None):
        merged_df = self.xbt_df.merge(other.xbt_df[['id'] + features_to_merge], on='id', how='outer')
        if fill_values:
            for f1, fv1 in fill_values.items():
                merged_df[f1][merged_df[f1].isna()] = fv1
        self.xbt_df = merged_df
        if feature_encoders:
            self._feature_encoders.update(feature_encoders)
        if target_encoders:
            self._target_encoders.update(target_encoders)
        if output_formatters:
            self._output_formatters.update(output_formatters)
            
    def filter_features(self, feature_list):
        """
        Create a XbtDataset with a subset of the columns in this data.
        """
        subset_df = self.xbt_df[feature_list] 
        filtered_dataset = XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
        filtered_dataset._feature_encoders = self._feature_encoders
        filtered_dataset._target_encoders = self._target_encoders
        return filtered_dataset
    
    def train_test_split(self, train_fraction=0.8, random_state=None, features=None, refresh=False, split_name=TRAIN_SET_FEATURE):
        """
        Create a train/test split for this dataset, and add a column to the dataframe which is True if that row
        is in the train set, and False if it is in the test set. If this column is already present, the existing
        labels are used to split the data, unless efresh is True which causes a new split to be calculated.
        """
        
        def construct_param_combos(xbt_df, features):
            current_feature = features[0]
            current_values = [f1 for f1 in xbt_df[current_feature].unique()]
            if len(features) == 1:
                return [[f1] for f1 in current_values]
            feature_tuples = construct_param_combos(xbt_df, features[1:])
            return [[f2] + f1 for f1 in feature_tuples for f2 in current_values]

        test_fraction = 1.0 - train_fraction
        
        
        if refresh or split_name not in self.xbt_df.columns:
            self.xbt_df[split_name] = True
            
            opt_dict = {'frac': test_fraction}
            if random_state:
                opt_dict['random_state'] = random_state
            
            if features:
                # if we want to ensure the split is even across features, we 
                # have to process each combination of features sepratately. 
                # One example might be splitting on year and instrument. We 
                # then have to go through each combination of possible year 
                # and instrument values, and sample from the subset of profiles 
                # that matches that year and instrument.
                for params in construct_param_combos(self.xbt_df, features):
                    self.xbt_df.loc[self._get_subset_df(dict(zip(features, params))).sample(**opt_dict).index,TRAIN_SET_FEATURE] = False
                
            else:
                self.xbt_df.loc[self.xbt_df.sample(**opt_dict).index,split_name] = False
            self.xbt_df[split_name] = self.xbt_df[split_name].astype(bool)
            
        # once we have a feature that divides the profiles into train and test, actally extract the 
        # train and test profiles as XbtDataset objects.
        xbt_df_train = self.xbt_df[self.xbt_df[split_name]]
        xbt_df_test = self.xbt_df[~self.xbt_df[split_name]]
        
        xbt_train = XbtDataset(year_range=self.year_range, directory=self.directory, df = xbt_df_train)
        xbt_train._feature_encoders = self._feature_encoders
        xbt_train._target_encoders = self._target_encoders
        xbt_test = XbtDataset(year_range=self.year_range, directory=self.directory, df = xbt_df_test)
        xbt_test._feature_encoders = self._feature_encoders
        xbt_test._target_encoders = self._target_encoders
        return (xbt_train, xbt_test)
            
        
    def generate_random_folds(num_folds, fold_feature_name):
        self.xbt_df[fold_feature_name] = numpy.round(numpy.random.random(xbt_labelled.shape[0])*num_folds) % num_folds
        self.xbt_df[fold_feature_name] = xbt_labelled.xbt_df[fold_feature_name].astype(int)
        
    def generate_folds_by_feature(self, feature_name, num_folds, fold_feature_name):
        """
        This function generate folds for train/test splitting, but instead of just generating them at random,
        it find all the unique values of particular features, divides the unique values into num_folds groups,
        then labels all observations according to the number of the group where value of the feature for that
        observation is in the group.
        """
        feature_values  = list(self.xbt_df[feature_name].unique())
        num_values = len(feature_values)
        df1 = pandas.DataFrame({feature_name : feature_values , 'fold' : (numpy.random.random_integers(low=0, high=num_folds-1, size=(num_values,))) })        
        self.xbt_df[fold_feature_name] = 0
        for ix1 in range(1,num_folds):
            fvl1 = list(df1[df1['fold'] == ix1][feature_name])
            self.xbt_df.loc[self.xbt_df[self.xbt_df[feature_name].apply(lambda x: x in fvl1)].index, fold_feature_name] = ix1
            
        self._feature_encoders[fold_feature_name] = get_cat_ml_feature(self.xbt_df[[fold_feature_name]], do_transform=False)
    
    def update_split_from_fold(split_feature, fold_feature, fold_num):
        self.xbt_df[split_feature] = self.xbt_df[fold_feature_name] != fold_num
        
    def _encode_dataset(self, encoders, features_to_process, processor_funcs, return_data):
        ml_features = {}
        column_indices = {}
        column_start = 0
            
        for f1 in features_to_process:
            create_encoder = False
            try: 
                encoder_f1 = encoders[f1]
                if return_data:    
                    mlf1 = encoder_f1.transform(self.xbt_df[[f1]])
                else:
                    mlf1 = None
            except KeyError:
                create_encoder = True
                mlf1 = None
            
            if create_encoder:
                try: 
                    (encoder, mlf1) = processor_funcs[f1](self.xbt_df[[f1]], do_transform=return_data)
                    encoders[f1] = encoder
                except KeyError:
                    raise RuntimeError(f'Attempting to preprocess unknown feature {f1}')
            
            if return_data:
                ml_features[f1] = mlf1
                column_indices[f1] = column_start
                if len(mlf1.shape) > 1:
                    column_start += mlf1.shape[1]
                else:
                    column_start += 1

        if return_data:
            ml_ds = numpy.concatenate([v1 for k1,v1 in ml_features.items()], axis=1)
            
        else:
            ml_ds = None
            
        return ml_ds, encoders, ml_features, column_indices

            
    def get_ml_dataset(self, refresh=False, return_data=True):
        """
        Get the data in the dataframe encoded for use bt a machine learning algorithm. This
        is done on a per feature, and then the outputs are combined into a numpy array.
        The enocding is delegated to the members of the FEATURE_PROCESSORS dictionary, which
        contains an element for each data column. The essentially create a suitable scikit learn
        object to encode the data e.g. OneHotEncoder for categorical data or MinMaxScaler for 
        latitude and longitude.
        
        This function can also be used to just create encoders and not transform the data. This is useful
        if you want consisent encoding of the data when looking at different subsets when not all values
        in category may be present in the subset. This owrks because when a subset is created by any of the
        subsetting functions (e.g. filter_obs or filter_features), the feature encoders dictionary is passed
        to the new class. This means you will get the same encoding in the parent and child objects, so results
        for different subsets can easily be compared.
        
        
        """
        if refresh:
            self._feature_encoders = {}
        features_to_process = self._get_features_to_process()
        
        ml_ds, encoders, ml_features, column_indices = self._encode_dataset(
            self._feature_encoders, features_to_process, FEATURE_PROCESSORS, return_data)
        
        self._feature_encoders.update(encoders)
        
        return (ml_ds, self._feature_encoders, ml_features, column_indices)
    
    def encode_target(self, refresh=False, return_data=True):
        """
        Get the data in the dataframe encoded for use bt a machine learning algorithm. This
        is done on a per feature, and then the outputs are combined into a numpy array.
        The enocding is delegated to the members of the FEATURE_PROCESSORS dictionary, which
        contains an element for each data column. The essentially create a suitable scikit learn
        object to encode the data e.g. OneHotEncoder for categorical data or MinMaxScaler for 
        latitude and longitude.
        
        This function can also be used to just create encoders and not transform the data. This is useful
        if you want consisent encoding of the data when looking at different subsets when not all values
        in category may be present in the subset. This owrks because when a subset is created by any of the
        subsetting functions (e.g. filter_obs or filter_features), the feature encoders dictionary is passed
        to the new class. This means you will get the same encoding in the parent and child objects, so results
        for different subsets can easily be compared.
        
        
        """
        if refresh:
            self._target_encoders = {}
        features_to_process = self._get_features_to_process()
        
        ml_ds, encoders, ml_features, column_indices = self._encode_dataset(
            self._target_encoders, features_to_process, TARGET_PROCESSORS, return_data)
        
        self._target_encoders.update(encoders)
        
        return (ml_ds, self._target_encoders, ml_features, column_indices)

    
    def get_cruise_stats(self):
        cruise_stats = {}
        cruise_id_list = self.cruises
        num_unknown_model = 0
        num_no_model_data = 0
        num_unknown_manufacturer = 0
        num_no_manufacturer_data = 0
        for cid in cruise_id_list:
            cruise_data = {}
            cruise_obs = self.filter_obs({KEY_DICT['CRUISE']: cid})
            cruise_data['num_obs'] = cruise_obs.shape[0]
            
            cruise_data['models'] = list(cruise_obs.models)
            cruise_data['num_models'] = len(cruise_data['models'])
            cruise_data['num_unknown_model'] = cruise_obs.num_unknown_model
            if cruise_data['num_unknown_model'] > 0:
                num_unknown_model += 1
            if cruise_data['num_unknown_model'] == cruise_data['num_obs']:
                num_no_model_data += 1
            
            cruise_data['manufacturers'] = list(cruise_obs.models)
            cruise_data['num_manufacturers'] = len(cruise_data['manufacturers'])
            cruise_data['num_unknown_manufacturer'] = cruise_obs.num_unknown_manufacturer
            if cruise_data['num_unknown_manufacturer'] > 0:
                num_unknown_manufacturer += 1
            if cruise_data['num_unknown_manufacturer'] == cruise_data['num_obs']:
                num_no_manufacturer_data += 1
            
            cruise_stats[cid] = cruise_data
        self.cruise_stats = pandas.DataFrame.from_dict(cruise_stats, orient='index')
        return (self.cruise_stats, 
                num_unknown_model, 
                num_no_model_data, 
                num_unknown_manufacturer, 
                num_no_manufacturer_data)
    
    @property
    def shape(self):
        return self.xbt_df.shape
    
    @property
    def data(self):
        return self.xbt_df
    
    @property
    def unknown_model_dataset(self):
        imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 1]
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        other_obs = no_imeta_obs[no_imeta_obs.model.apply(lambda x: check_value_found(UNKNOWN_STR, x))]
        subset_df = pandas.concat([imeta_obs, other_obs])
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
    
    @property
    def unknown_manufacturer_dataset(self):
        imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 1]
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        other_obs = no_imeta_obs[no_imeta_obs.manufacturer.apply(lambda x: check_value_found(UNKNOWN_STR, x))]
        subset_df = pandas.concat([imeta_obs, other_obs])
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
                                
    @property
    def known_model_dataset(self):
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        subset_df = no_imeta_obs[no_imeta_obs.model.apply(lambda x: not check_value_found(UNKNOWN_STR, x))]
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
    
    @property
    def known_manufacturer_dataset(self):
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        subset_df = no_imeta_obs[no_imeta_obs.manufacturer.apply(lambda x: not check_value_found(UNKNOWN_STR, x))]
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)

    @property
    def num_unknown_model(self):
        imeta_num = sum(self.xbt_df.imeta_applied)
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        other_unknown = sum(no_imeta_obs.model.apply(lambda x: check_value_found(UNKNOWN_STR, x)))
        return imeta_num + other_unknown
    
    
    @property
    def num_unknown_manufacturer(self):
        imeta_num = sum(self.xbt_df.imeta_applied)
        no_imeta_obs = self.xbt_df[self.xbt_df.imeta_applied == 0]
        other_unknown = sum(no_imeta_obs.manufacturer.apply(lambda x: check_value_found(UNKNOWN_STR, x)))
        return imeta_num + other_unknown
    
    @property
    def instruments_by_platform(self):
        return self.get_atrributes_per_subset('platform', 'instrument')
        
    @property
    def platforms_by_instrument(self):
        return self.get_atrributes_per_subset('instrument', 'platform')
            
    def get_atrributes_per_subset(self, subset, attr):
        values = self.get_property_values(subset)
        return {
            v1: list(self.filter_obs({subset: v1})[attr].unique())
            for v1 in values }
    


    @property
    def num_obs(self):
        return self.xbt_df.shape[0]
    
    def get_property_values(self, key):
        return list(self.xbt_df[key].unique())
    
    @property
    def cruises(self):
        return list(self.xbt_df.cruise_number.unique())

    @property
    def instruments(self):
        return list(self.xbt_df.instrument.unique())
    
    @property
    def countries(self):
        return list(self.xbt_df.country.unique())

    @property
    def platforms(self):
        return list(self.xbt_df.platform.unique())
    
    @property
    def institutes(self):
        return list(self.xbt_df.institute.unique())
    
    @property
    def manufacturers(self):
        return list(self.xbt_df.manufacturer.unique())
    
    @property
    def models(self):
        return list(self.xbt_df.model.unique())
    
    @property
    def instrument_distribution(self):
        return self._calc_distribution('instruments')

    @property
    def platform_distribution(self):
        return self._calc_distribution('platforms')
    
    def get_distribution(self, var_name):
        return {v1: self.xbt_df[self.xbt_df[var_name] == v1].shape[0] 
                for v1 in self.get_property_values(var_name)}
    
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            return getattr(self.xbt_df, key)
        return None
        

def get_data_stats(file_path, year):
    print(f'processing year {year} from file {file_path}')
    dataset1 = XbtDataset(file_path)
    return (year,
            dataset1.num_obs,
             dataset1.num_unknown_brand_or_type,
             dataset1.num_unknown_type,
             dataset1.instruments,
             dataset1.platforms,
             )
