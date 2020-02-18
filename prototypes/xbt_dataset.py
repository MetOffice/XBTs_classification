import os
import re
import pandas

XBT_FNAME_TEMPLATE = 'xbt_{year}.csv'

INSTRUMENT_REGEX_STRING = 'XBT[:][\s](?P<model>[\w\s;:-]+)([\s]*)([(](?P<manufacturer>[\w\s.:;-]+)[)])?'
REGEX_MANUFACTURER_GROUP = 'manufacturer'
REGEX_MODEL_GROUP = 'model'

UNKOWN_MODEL_STR = 'TYPE UNKNOWN'
UNKOWN_MANUFACTURER_ST = 'UNKNOWN BRAND'

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
    
   
    def get_cruise_stats(self):
        cruise_number_list = list(set(self.xbt_df['cruise_number'].values))
        self.cruise_stats = {cid: {
            'num_obs': self.xbt_df[self.xbt_df['cruise_number'] == cid].shape[
                0],
            'instruments':
                list(self.xbt_df[
                         self.xbt_df[
                             'cruise_number'] == cid].instrument.unique()),
            'num_instruments':
                len(self.xbt_df[
                        self.xbt_df[
                            'cruise_number'] == cid].instrument.unique()), }
            for cid in cruise_number_list}
        return self.cruise_stats

    @property
    def unknown_model_dataset(self):
        return self.xbt_df[self.xbt_df.model.str == UNKOWN_MODEL_STR]
    
    def unknown_manufacturer_dataset(self):
        return self.xbt_df[self.xbt_df.manufacturer.str == UNKOWN_MANUFACTURER_ST]
    

    @property
    def num_unknown_model(self):
        return self.unknown_type_dataset.shape[0]

    @property
    def num_obs(self):
        return self.xbt_df.shape[0]

    @property
    def instruments(self):
        return list(self.xbt_df.instrument.unique())

    @property
    def platforms(self):
        return list(self.xbt_df.platform.unique())

    @property
    def instrument_distribution(self):
        return {ins1: self.xbt_df[self.xbt_df.instrument == ins1].shape[0] for ins1 in
                self.instruments}

    @property
    def platform_distribution(self):
        return {plat1: self.xbt_df[self.xbt_df.platform == plat1].shape[0] for plat1 in
                self.platforms}


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
