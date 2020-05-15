import os
import re
import pandas
import functools
import datetime
import dask.dataframe
import numpy
import sklearn.preprocessing

XBT_FNAME_TEMPLATE = 'xbt_{year}.csv'

INSTRUMENT_REGEX_STRING = 'XBT[:][\s](?P<model>[\w\s;:-]+)([\s]*)([(](?P<manufacturer>[\w\s.:;-]+)[)])?'
REGEX_MANUFACTURER_GROUP = 'manufacturer'
REGEX_MODEL_GROUP = 'model'

UNKNOWN_STR = 'UNKNOWN'
UNKNOWN_MODEL_STR = 'TYPE UNKNOWN'
UNKNOWN_MANUFACTURER_STR = 'UNKNOWN BRAND'

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

def get_year(dt_str):
    try:
        dt1 = datetime.datetime.strptime(dt_str,'%Y%m%d')
        year = dt1.year
        month = dt1.month
        day = dt1.day
    except ValueError:
        year = 0
        month=0
        day = 0
    return year, month, day

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
    )
    encoder.fit(cat_feature)
    if do_transform:
        ml_feature = encoder.transform(cat_feature)
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


CATEGORICAL_LOADED_FEATURES = [
    'country', 'institute', 'platform', 'cruise_number', 
    'instrument', 'imeta_applied'
]
CATEGORICAL_GENERATED_FEATURES = [
    'model', 'manufacturer',
]
CATEGORICAL_FEATURES = CATEGORICAL_LOADED_FEATURES + CATEGORICAL_GENERATED_FEATURES
PROFILE_FEATURES = ['temperature_profile', 'depth_profile']
QUALITY_FLAG_FEATURES = ['temperature_quality_flag', 'depth_quality_flag']
ID_FEATURES = ['id']
ORDINAL_FEATURES = ['year']
MINMAX_FEATURES = ['max_depth']
NORMAL_DIST_FEATURES = []
CUSTOM_FEATURES = ['lat', 'lon']
OTHER_FEATURES = ['date']
XBT_CONVERTERS = {'temperature_profile': eval,
                  'depth_profile': eval,
                 }


FEATURE_PROCESSORS = {}
FEATURE_PROCESSORS.update({f1: get_cat_ml_feature for f1 in CATEGORICAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_ord_ml_feature for f1 in ORDINAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_minmax_ml_feature for f1 in MINMAX_FEATURES})
FEATURE_PROCESSORS.update({f1: get_num_ml_feature for f1 in NORMAL_DIST_FEATURES})
FEATURE_PROCESSORS.update({'lat': functools.partial(get_minmaxfixed_ml_feature, -90.0, 90.0),
                           'lon': functools.partial(get_minmaxfixed_ml_feature, -180.0, 180.0),
                           'max_depth': functools.partial(get_minmaxfixed_ml_feature, 0.0, 2000.0),
                          })

TRAIN_SET_FEATURE = 'training_set'
ML_EXCLUDE_LIST = ['date', 'id', 'month', 'day', TRAIN_SET_FEATURE]

def read_csv(fname, features_to_load, converters=None):
#     print(f'reading {fname}')
    return pandas.read_csv(fname,
                           converters=converters,
                           usecols=features_to_load
                          )

def do_concat(df_list, axis=1, ignore_index=True):
    xbt_df = pandas.concat(df_list, ignore_index=ignore_index)
    for feature1 in CATEGORICAL_FEATURES:
                xbt_df[feature1] = xbt_df[feature1].astype('category') 
    return xbt_df


def do_preprocessing(xbt_df):
    # Model and manufacturer are stored in the CSV file as a single string called "instrument type", separating into 
    # seprate columns for learning these separately 
    xbt_df['model'] = xbt_df.instrument.apply(get_model)
    xbt_df['manufacturer'] = xbt_df.instrument.apply(get_manufacturer)
    date_columns = ['year','month','day']
    date_elements = pandas.DataFrame(list(xbt_df.date.apply(str).apply(get_year)), columns=date_columns)
    for col1 in date_columns:
        xbt_df[col1] = date_elements[col1]
    # exclude bad dates
    xbt_df = xbt_df[xbt_df['year'] != 0]
    # exclude bad depths
    xbt_df = xbt_df[xbt_df['max_depth'] < 2000.0]
    xbt_df = xbt_df[xbt_df['max_depth'] > 0.0]

    return xbt_df


class XbtDataset():
    def __init__(self, directory, year_range, df=None, use_dask=False, load_profiles=False, load_quality_flags=False):
        self._use_dask = use_dask
        self._load_profiles = load_profiles
        self._load_quality_flags = load_quality_flags
        self.features_to_load = CATEGORICAL_LOADED_FEATURES + MINMAX_FEATURES + CUSTOM_FEATURES + ID_FEATURES + OTHER_FEATURES
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
        self.directory = directory
        self.year_range = year_range
        self.dataset_files = [os.path.join(self.directory, XBT_FNAME_TEMPLATE.format(year=year)) for year in range(year_range[0], year_range[1]+1)]
        self.xbt_df = df
        self._feature_encoders = {}
        if self.xbt_df is None:
            self._load_data() 

    def _load_data(self):
        print(f'load the following features: {self.features_to_load}')
        df_in_list = [self._read_func(year_csv_path, self.features_to_load) for year_csv_path in self.dataset_files]
        df_processed = [self._preproc_func(df_in) for df_in in df_in_list]
        self.xbt_df = self._concat_func(df_processed)
    
    def filter_obs(self, key, value):
        subset_df = self.xbt_df 
        if key == 'labelled':
            if value == 'labelled':
                print('extracting labelled')
                subset_df = self.xbt_df[self.xbt_df.instrument.apply(lambda x: not check_value_found(UNKNOWN_STR, x))] 
                subset_df = subset_df[subset_df.imeta_applied == 0]
            elif value == 'unlabelled':
                print('extracting unlabelled')
                subset_df1 = self.xbt_df[self.xbt_df.instrument.apply(lambda x: check_value_found(UNKNOWN_STR, x))] 
                subset_df2 = self.xbt_df[self.xbt_df.imeta_applied == 1] 
                subset_df = pandas.concat([subset_df1, subset_df2], axis=0)
            elif value == 'imeta':
                print('extracting profiles with intelligent metadata algorithm applied')
                subset_df = self.xbt_df[self.xbt_df.imeta_applied == 1] 
            elif value == 'all':
                subset_df = self.xbt_df
        else:
            try:
                subset_df = subset_df[subset_df[key].apply(lambda x: value in x)]
            except TypeError:
                subset_df = subset_df[subset_df[key] == value]
        filtered_dataset = XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
        filtered_dataset._feature_encoders = self._feature_encoders                
        return filtered_dataset
    
    def filter_features(self, feature_list):
        subset_df = self.xbt_df[feature_list] 
        filtered_dataset = XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
        filtered_dataset._feature_encoders = self._feature_encoders
        return filtered_dataset
    
    def train_test_split(self, train_fraction=0.8, random_state=None, split_on_feature=None, refresh=False):
        test_fraction = 1.0 - train_fraction
        if refresh or TRAIN_SET_FEATURE not in self.xbt_df.columns:
            opt_dict = {'frac': test_fraction}
            if random_state:
                opt_dict['random_state'] = rand
            self.xbt_df[TRAIN_SET_FEATURE] = True
            if split_on_feature:
                for year in range(*self.year_range):
                    self.xbt_df.loc[self.xbt_df[self.xbt_df.year == year].sample(**opt_dict).index,TRAIN_SET_FEATURE] = False
            else:
                self.xbt_df.loc[self.xbt_df.sample(**opt_dict).index,TRAIN_SET_FEATURE] = False
            
        xbt_df_train = self.xbt_df[self.xbt_df[TRAIN_SET_FEATURE]]
        xbt_df_test = self.xbt_df[self.xbt_df[TRAIN_SET_FEATURE].apply(lambda x: not x)]
        xbt_train = XbtDataset(year_range=self.year_range, directory=self.directory, df = xbt_df_train)
        xbt_train._feature_encoders = self._feature_encoders
        xbt_test = XbtDataset(year_range=self.year_range, directory=self.directory, df = xbt_df_test)
        xbt_test._feature_encoders = self._feature_encoders
        return (xbt_train, xbt_test)
            
        
    def get_ml_dataset(self, refresh=False, return_data=True):
        ml_features = {}
        if refresh:
            self._feature_encoders = {}
        column_indices = {}
        column_start = 0
        features_to_process = [feature1 for feature1 in self.xbt_df.columns 
                               if feature1 not in ML_EXCLUDE_LIST]
        for f1 in features_to_process:
            create_encoder = False
            try: 
                mlf1 = self._feature_encoders[f1].transform(self.xbt_df[[f1]])
            except KeyError:
                create_encoder = True
            
            if create_encoder:
                try: 
                    (encoder, mlf1) = FEATURE_PROCESSORS[f1](self.xbt_df[[f1]], do_transform=return_data)
                    self._feature_encoders[f1] = encoder
                except KeyError:
                    raise RuntimeError(f'Attempting to preprocess unknown feature {f1}')
            
            if return_data:
                ml_features[f1] = mlf1
                column_indices[f1] = column_start
                column_start += mlf1.shape[1]
                
        if return_data:
            ml_ds = numpy.concatenate([v1 for k1,v1 in ml_features.items()], axis=1)
        else:
            ml_ds = None
        return (ml_ds, self._feature_encoders, ml_features, column_indices)
    
    
    def get_cruise_stats(self):
        cruise_stats = {}
        cruise_id_list = self.cruises
        num_unknown_model = 0
        num_no_model_data = 0
        num_unknown_manufacturer = 0
        num_no_manufacturer_data = 0
        for cid in cruise_id_list:
            cruise_data = {}
            cruise_obs = self.filter_obs(KEY_DICT['CRUISE'], cid)
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
            v1: list(self.filter_obs(subset, v1)[attr].unique())
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
