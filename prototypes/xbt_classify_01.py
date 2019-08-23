#!/usr/bin/env python

import os
import sklearn.model_selection
import sklearn.neighbors
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.neural_network
import pandas

import pdb
import xbt_preprocess
from xbt_preprocess import XBT_RESULTS_DIR, XBT_PATH_TEMPLATE


NUM_CV_SCORES = 3


# INSTRUMENT_REGEX_STRING2 = 'XBT\: (?P<type>[\w\s\-;:]+)([ ][(](?P<brand>[\w .-:;]+)[)])'


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

    xbt_df = xbt_preprocess.load_and_preprocess_data(path_xbt_data)

    selected_input = ['cruise_number', 'institute', 'platform', 'country',
                      'lat', 'lon']
    input_features = {sf1: xbt_preprocess.INPUT_FEATURE_PROCESSORS[sf1] for sf1
                      in selected_input}

    output_features = ['type', 'brand']
    X, y, X_train, X_test, y_train, y_test = xbt_preprocess.get_ml_dataset(
        xbt_df, input_features, output_features[0])

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

    # Classification used multilayer perceptron neural et classifier
    classifiers['mlp'] = {
        'classifier': sklearn.neural_network.MLPClassifier(
            solver='lbfgs',
            alpha=1e-5,
            hidden_layer_sizes=(5, 2),
            random_state=1),
        'name': 'MLP Neural Network Classifier',
        'alpha': 1e-5,
        'hidden_layer_sizes': (5, 2),
        'random_state': 1,
    }

    for class_name, class_dict in classifiers.items():
        print(f'trying classifier {class_name}')
        # run_classifier(class_dict, X_train, y_train, X_test, y_test)
        classifiers[class_name]['score'] = sklearn.model_selection.cross_val_score(class_dict['classifier'], X, y, cv=NUM_CV_SCORES)

    return classifiers

def main():
    year_range = (1988,1998)
    year_list = range(*year_range)
    results_raw  = {}

    results_for_output = []
    for year in year_list:
        print(f'processing year {year:04d}')

        results_raw[year] = process_year(year)
        for model_id, model1 in results_raw[year].items():
            res1 = {'model': model1['name'], 'year': year}
            for ix1, score_val in enumerate(model1['score']):
                res1['score_{0}'.format(ix1)] = score_val
            results_for_output += [res1]

    results_df = pandas.DataFrame(results_for_output)
    res_fname = f'classification_scores_xbt_{year_range[0]}_{year_range[1]}.csv'
    res_path = os.path.join(XBT_RESULTS_DIR, res_fname)
    results_df.to_csv(res_path)

if __name__ == '__main__':
    main()
