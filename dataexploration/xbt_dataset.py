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

def normalise_lat(feature_lat):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.data_min_ = -90.0
    encoder.data_max_ = 90.0
    ml_feature = encoder.transform(feature_lat)
    return (encoder, ml_feature)


def normalise_lon(feature_lon):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.data_min_ = -180.0
    encoder.data_max_ = 180.0
    ml_feature = encoder.transform(feature_lon)

def get_cat_ml_feature(cat_feature):
    encoder = sklearn.preprocessing.OneHotEncoder()
    encoder.fit(cat_feature)
    ml_feature = encoder.transform(cat_feature).toarray()
    return (encoder, ml_feature)

def get_ord_ml_feature(ord_feature):
    encoder = sklearn.preprocessing.OrdinalEncoder()
    encoder.fit(ord_feature)
    ml_feature = encoder.transform(ord_feature)
    return (encoder, ml_feature)

def get_num_ml_feature(numerical_feature):
    encoder = sklearn.preprocessing.StandardScaler()
    encoder.fit(numerical_feature)
    ml_feature = encoder.transform(numerical_feature)
    return (encoder, ml_feature)

def get_minmaxfixed_ml_feature(data_min, data_max, minmax_feature):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.fit(minmax_feature)
    encoder.data_min_ = data_min
    encoder.data_max_ = data_max
    ml_feature = encoder.transform(minmax_feature)
    return (encoder, ml_feature)

def get_minmax_ml_feature(minmax_feature):
    encoder = sklearn.preprocessing.MinMaxScaler()
    encoder.fit(minmax_feature)
    ml_feature = encoder.transform(minmax_feature)
    return (encoder, ml_feature)


CATEGORICAL_FEATURES = ['country', 'institute', 'platform', 'cruise_number', 
                        'instrument', 'temperature_quality_flag', 'model', 
                        'manufacturer',
                       ]
ORDINAL_FEATURES = ['year']
MINMAX_FEATURES = []
NORMAL_DIST_FEATURES = []
CUSTOM_FEATURES = ['lat', 'lon']

FEATURE_PROCESSORS = {}
FEATURE_PROCESSORS.update({f1: get_cat_ml_feature for f1 in CATEGORICAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_ord_ml_feature for f1 in ORDINAL_FEATURES})
FEATURE_PROCESSORS.update({f1: get_minmax_ml_feature for f1 in MINMAX_FEATURES})
FEATURE_PROCESSORS.update({f1: get_num_ml_feature for f1 in NORMAL_DIST_FEATURES})
FEATURE_PROCESSORS.update({'lat': functools.partial(get_minmaxfixed_ml_feature, -90.0, 90.0),
                           'lon': functools.partial(get_minmaxfixed_ml_feature, -180.0, 180.0),
                           'max_depth': functools.partial(get_minmaxfixed_ml_feature, 0.0, 2000.0),
                          })

class XbtDataset():
    def __init__(self, directory, year_range, df=None, use_dask=False):
        self._use_dask = use_dask
        if self._use_dask:
            self._read_func = dask.dataframe.read_csv
        else:
            self._read_func = pandas.read_csv
        self.directory = directory
        self.year_range = year_range
        self.dataset_files = [os.path.join(self.directory, XBT_FNAME_TEMPLATE.format(year=year)) for year in range(year_range[0], year_range[1])]
        self.xbt_df = df
        if self.xbt_df is None:
            self._load_data() 

    def _load_data(self):
        
        self.xbt_df = pandas.concat((self._read_func(year_csv_path).drop(EXCLUDE_LIST, axis=1) 
                                     for year_csv_path in self.dataset_files),
                                    ignore_index=True)
        
        # Model and manufacturer are stored in the CSV file as a single string called "instrument type", separating into 
        # seprate columns for learning these separately 
        self.xbt_df['model'] = self.xbt_df.instrument.apply(get_model)
        self.xbt_df['manufacturer'] = self.xbt_df.instrument.apply(get_manufacturer)
        date_columns = ['year','month','day']
        date_elements = pandas.DataFrame(list(self.xbt_df.date.apply(str).apply(get_year)), columns=date_columns)
        for col1 in date_columns:
            self.xbt_df[col1] = date_elements[col1]
        # exclude bad dates
        self.xbt_df = self.xbt_df[self.xbt_df['year'] != 0]
        for feature1 in CATEGORICAL_FEATURES:
            self.xbt_df[feature1] = self.xbt_df[feature1].astype('category')
    
    def filter_obs(self, key, value):
        subset_df = self.xbt_df 
        if key == 'labelled':
            if value == 'labelled':
                print('extracting labelled')
                subset_df = self.xbt_df[self.xbt_df.instrument.apply(lambda x: not check_value_found(UNKNOWN_STR, x))] 
            elif value == 'unlabelled':
                print('extracting unlabelled')
                subset_df = self.xbt_df[self.xbt_df.instrument.apply(lambda x: check_value_found(UNKNOWN_STR, x))] 
            elif value == 'all':
                subset_df = self.xbt_df
        else:
            try:
                subset_df = subset_df[subset_df[key].apply(lambda x: value in x)]
            except TypeError:
                subset_df = subset_df[subset_df[key] == value]
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
    
    def filter_features(self, feature_list):
        subset_df = self.xbt_df[feature_list]     
        return XbtDataset(year_range=self.year_range, directory=self.directory, df = subset_df)
    
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
        return self.xbt_df[self.xbt_df.model.apply(
            functools.partial(check_value_found, UNKNOWN_STR))]        
    
    @property
    def unknown_manufacturer_dataset(self):
        return self.xbt_df[self.xbt_df.manufacturer.apply(
            functools.partial(check_value_found, UNKNOWN_STR))]        

    @property
    def known_model_dataset(self):
        return self.xbt_df[-self.xbt_df.model.apply(
            functools.partial(check_value_found, UNKNOWN_STR))]        
    
    @property
    def known_manufacturer_dataset(self):
        return self.xbt_df[-self.xbt_df.manufacturer.apply(
            functools.partial(check_value_found, UNKNOWN_STR))]        

    @property
    def num_unknown_model(self):
        return self.unknown_model_dataset.shape[0]
    
    @property
    def num_unknown_manufacturer(self):
        return self.unknown_manufacturer_dataset.shape[0]
    
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
        
        
    def get_ml_dataset(self):
        ml_features = []
        encoders = []
        for f1 in self.xbt_df.columns:
            try: 
                (encoder, mlf1) = FEATURE_PROCESSORS[f1](self.xbt_df[[f1]])
            except KeyError:
                raise RuntimeError(f'Attempting to preprocess unknown feature {f1}')
            ml_features += [mlf1]
            encoders += [encoder]
        ml_ds = numpy.concatenate(ml_features, axis=1)
        return (ml_ds, encoders)


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
