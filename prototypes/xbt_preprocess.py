#!/usr/bin/env python

import re
import os
import numpy
import pandas
import sklearn.model_selection
import argparse

XBT_AVAILABLE_YEARS = (1966, 2016)

XBT_BASE_DIR = '/data/users/shaddad/xbt-data/'
XBT_PATH_TEMPLATE = os.path.join(XBT_BASE_DIR, 'annual_csv',
                                 'xbt_{year:04d}.csv')
XBT_RESULTS_DIR = os.path.join(XBT_BASE_DIR, 'results')

INSTRUMENT_REGEX_STRING = 'XBT\: (?P<type>[\w\s;:-]+)[\s]([(](?P<brand>[\w -.:;]+)[)])?'

INSTRUMENT_REGEX_STRING3 = 'XBT[:][\s](?P<type>[\w\s;:-]+)([\s]*)([(](?P<brand>[\w\s.:;-]+)[)])?'

import pdb


def category_to_label(feature1):
    return numpy.array(pandas.get_dummies(feature1))


def normalise_lat(feature_lat):
    return numpy.array(feature_lat / 90.0).reshape(-1, 1)


def normalise_lon(feature_lon):
    return numpy.array(feature_lon / 180.0).reshape(-1, 1)


INPUT_FEATURE_PROCESSORS = {'cruise_number': category_to_label,
                            'institute': category_to_label,
                            'platform': category_to_label,
                            'country': category_to_label,
                            'lat': normalise_lat,
                            'lon': normalise_lon}


def get_type(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING3, instr_str)
        type_str = matches.group('type')
    except AttributeError as e1:
        pdb.set_trace()
    return str(type_str)


def get_brand(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING3, instr_str)
        brand_str = matches.group('brand')
    except AttributeError as e1:
        pdb.set_trace()
    return str(brand_str)


def load_and_preprocess_data(path_xbt_data):
    xbt_df_raw = pandas.read_csv(path_xbt_data)[:10000]
    xbt_df = pandas.DataFrame.copy(
        xbt_df_raw[~xbt_df_raw.instrument.str.contains('UNKNOWN')])

    xbt_df['type'] = xbt_df.instrument.apply(get_brand)
    xbt_df['brand'] = xbt_df.instrument.apply(get_type)

    cat_features = ['country', 'cruise_number', 'institute', 'platform',
                    'type', 'brand']
    label_mappings = {}

    for feature_name in cat_features:
        try:
            label_enc1 = sklearn.preprocessing.LabelEncoder()
            label_enc1.fit(xbt_df[feature_name])
            xbt_df[feature_name] = label_enc1.transform(xbt_df[feature_name])
            label_mappings[feature_name] = label_enc1
        except:
            pdb.set_trace()

    return xbt_df


def get_ml_dataset(xbt_df, input_features, output_feature):
    input_features_data = {}

    for ifname, iffunc in input_features.items():
        input_features_data[ifname] = iffunc(xbt_df[ifname])

    X = numpy.concatenate(list(input_features_data.values()), axis=1)
    y = xbt_df[output_feature]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y)
    return X, y, X_train, X_test, y_train, y_test


def parse_args():
    parser1 = argparse.ArgumentParser(
        'preprocess XBT for use as input to a classification algorithm')
    help_msg = 'The start year of the data range to include in the preprocessing.'
    parser1.add_argument('--start-year', type=int, help=help_msg,
                         choices=range(*XBT_AVAILABLE_YEARS))

    help_msg = 'The end year of the data range to include in the preprocessing.'
    parser1.add_argument('--end-year', type=int, help=help_msg,
                         choices=range(*XBT_AVAILABLE_YEARS))
    help_msg = 'list of input arguments'
    parser1.add_argument('--inputs', type=str, nargs='+', help=help_msg)
    help_msg = 'The output feature to be targeted for classification'
    parser1.add_argument('--target-feature', type=str, help=help_msg)
    help_msg = 'The path to write the output csv file to.'
    parser1.add_argument('--output-path', type=str, help=help_msg)
    args1 = parser1.parse_args()
    return args1


def main():
    args1 = parse_args()
    xbt_year_list = []
    for year1 in range(args1.start_year, args1.end_year):
        print(f'loading year {year1}')
        path_year = XBT_PATH_TEMPLATE.format(year=year1)
        xbt_year = load_and_preprocess_data(path_year)
        xbt_year = xbt_year[args1.inputs + [args1.target_feature]]
        xbt_year_list += [xbt_year]
    pdb.set_trace()
    xbt_df = pandas.concat(xbt_year_list)
    xbt_df.to_csv(args1.output_path)


if __name__ == '__main__':
    main()
