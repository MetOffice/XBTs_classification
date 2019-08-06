import re

import numpy
import sklearn.model_selection
import sklearn.neighbors
import sklearn.ensemble
import sklearn.preprocessing
import pandas

import pdb

INSTRUMENT_REGEX_STRING = 'XBT\: (?P<type>[\w\s;:-]+)[\s]([(](?P<brand>[\w -.:;]+)[)])?'

INSTRUMENT_REGEX_STRING3 = 'XBT[:][\s](?P<type>[\w\s;:-]+)([\s]*)([(](?P<brand>[\w\s.:;-]+)[)])?'

XBT_PATH_TEMPLATE = '/data/users/shaddad/xbt-data/annual_csv/xbt_{year:04d}.csv'

# INSTRUMENT_REGEX_STRING2 = 'XBT\: (?P<type>[\w\s\-;:]+)([ ][(](?P<brand>[\w .-:;]+)[)])'

def get_type(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING3 ,instr_str)
        type_str = matches.group('type')
    except AttributeError as e1:
        pdb.set_trace()
    return str(type_str)

def get_brand(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING3,instr_str)
        brand_str = matches.group('brand')
    except AttributeError as e1:
        pdb.set_trace()
    return str(brand_str)

def category_to_label(feature1):
    return numpy.array(pandas.get_dummies(feature1))

def normalise_lat(feature_lat):
    return numpy.array(feature_lat / 90.0).reshape(-1,1)

def normalise_lon(feature_lon):
    return numpy.array(feature_lon / 180.0).reshape(-1, 1)

def load_and_preprocess_data(path_xbt_data):

    xbt_df_raw = pandas.read_csv(path_xbt_data)[:10000]
    xbt_df = pandas.DataFrame.copy(xbt_df_raw[~xbt_df_raw.instrument.str.contains('UNKNOWN')])

    xbt_df['type'] = xbt_df.instrument.apply(get_brand)
    xbt_df['brand'] = xbt_df.instrument.apply(get_type)

    cat_features = ['country','cruise_number','institute','platform','type','brand']
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

def run_classifier(classifier_dict, X_train, y_train, X_test, y_test):
    classifier = classifier_dict['classifier']
    print('run test for {0}'.format(classifier_dict['name']))
    classifier.fit(X_train, y_train)
    classifier.score(X_train, y_train)
    classifier.score(X_test, y_test)
    y_train_out = classifier.predict(X_train)
    y_test_out = classifier.predict(X_test)
    return y_train_out, y_test_out


def process_year(year):
    path_xbt_data = XBT_PATH_TEMPLATE.format(year=year)

    xbt_df = load_and_preprocess_data(path_xbt_data)

    input_features = {'cruise_number': category_to_label,
                      'institute': category_to_label,
                      'platform': category_to_label,
                      'country': category_to_label,
                      'lat': normalise_lat,
                      'lon': normalise_lon}

    output_features = ['type','brand']
    X, y, X_train, X_test, y_train, y_test = get_ml_dataset(xbt_df, input_features,
                                                      output_features[0])

    classifiers = {}

    n_neighbours = 10
    weights = 'distance'
    classifiers['nn10'] = {
        'classifier': sklearn.neighbors.KNeighborsClassifier(n_neighbours,
                                                             weights=weights),
        'name': 'NN-10',
        'n_neighours': n_neighbours,
        'weights': weights,
    }

    n_neighbours = 5
    weights = 'distance'
    classifiers['nn5'] = {
        'classifier': sklearn.neighbors.KNeighborsClassifier(n_neighbours,
                                                             weights=weights),
        'name': 'NN-5',
        'n_neighours': n_neighbours,
        'weights': weights,
    }

    classifiers['rf'] = {'classifier': sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0),
                         'name': 'random_forest'
                         }
    for class_name, class_dict in classifiers.items():
        # run_classifier(class_dict, X_train, y_train, X_test, y_test)
        classifiers[class_name]['score'] = sklearn.model_selection.cross_val_score(class_dict['classifier'], X, y, cv=3)

    return classifiers

year_list = range(1988,1998)
results  = {}
for year in year_list:
    print(f'processing year {year:04d}')
    results[year] = process_year(year)









