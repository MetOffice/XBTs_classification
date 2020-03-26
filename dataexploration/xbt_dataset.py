import os
import re
import pandas

XBT_FNAME_TEMPLATE = 'xbt_{year}.csv'

INSTRUMENT_REGEX_STRING = 'XBT[:][\s](?P<model>[\w\s;:-]+)([\s]*)([(](?P<manufacturer>[\w\s.:;-]+)[)])?'
REGEX_MANUFACTURER_GROUP = 'manufacturer'
REGEX_MODEL_GROUP = 'model'

UNKOWN_MODEL_STR = 'TYPE UNKNOWN'
UNKOWN_MANUFACTURER_ST = 'UNKNOWN BRAND'

KEY_DICT = {
    'CRUISE': 'cruise_number',
}

def get_model(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        type_str = matches.group(REGEX_MODEL_GROUP)
    except AttributeError as e1:
        pass
    return str(type_str)


def get_manufacturer(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        brand_str = matches.group(REGEX_MANUFACTURER_GROUP)
    except AttributeError as e1:
        pass
    return str(brand_str)

class XbtDataset():
    def __init__(self, directory, year_range):
        self.directory = directory
        self.year_range = year_range
        self.dataset_files = [os.path.join(self.directory, XBT_FNAME_TEMPLATE.format(year=year)) for year in range(year_range[0], year_range[1])]
        self.xbt_df = None
        self._load_data()

    def _load_data(self):
        self.xbt_df = pandas.concat(pandas.read_csv(year_csv_path) for year_csv_path in self.dataset_files)
        # Model and manufacturer are stored in the CSV file as a single string called "instrument type", separating into 
        # seprate columns for learning these separately 
        self.xbt_df['model'] = self.xbt_df.instrument.apply(get_model)
        self.xbt_df['manufacturer'] = self.xbt_df.instrument.apply(get_manufacturer)
    
    def filter_obs(self, key, value):
        return self.xbt_df[self.xbt_df[key] == value]
    
    def get_cruise_stats(self):
        cruise_stats = {}
        for cid in self.cruises:
            cruise_data = {}
            cruise_obs = self.filter_obs(KEY_DICT['CRUISE'], cid)
            cruise_data['num_obs'] = cruise_obs.shape[0]
            cruise_data['instruments'] = list(cruise_obs.instrument.unique())
            cruise_data['num_instruments'] = len(cruise_data['instruments'])
            cruise_stats[cid] = cruise_data
        self.cruise_stats = pandas.DataFrame.from_dict(cruise_stats, orient='index')
        return self.cruise_stats

    @property
    def unknown_model_dataset(self):
        return self.xbt_df[self.xbt_df.model.str == UNKOWN_MODEL_STR]
    
    @property
    def unknown_manufacturer_dataset(self):
        return self.xbt_df[self.xbt_df.manufacturer.str == UNKOWN_MANUFACTURER_ST]
    
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
    def num_unknown_model(self):
        return self.unknown_type_dataset.shape[0]

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
    
    def get_distibution(self, var_name):
        return {v1: self.xbt_df[self.xbt_df[var_name] == v1].shape[0] for v1 in
                self[var_name]}
    
    def __getitem__(self, key):
        return getattr(self, key)
        
        


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
