"""
This algorithm for detemining the model and manufacturer is taken from the following paper by Palmer et. al.
https://journals.ametsoc.org/doi/full/10.1175/JTECH-D-17-0129.1
"""
import os
import datetime
import functools
import argparse

import dataexploration.xbt_dataset

def get_depth_category(depth_list, depth_value):
    previous = 0.0
    for ix1, current in enumerate(depth_list):
        if depth_value > previous and depth_value <= current:
            return ix1
        previous = current
    return None

def get_model_by_date(model1, model2, date_threshold, obs_date):
    if obs_date < date_threshold:
        return model1
    return model2


TSK_countries = ['JAPAN', 'CHINA', 'TAIWAN', 'KOREA']
MANUFACTURERS = ['SIPPICAN', 'TSK - TSURUMI SEIKI Co.']
XBT_MAX_DEPTH = 2000.0
IMETA_VALUES = { 'SIPPICAN' : { 'depths': [360.0,600.0,1000.0,1350.0,2000.0],
                               'models': [functools.partial(get_model_by_date, 'T4','T10', datetime.datetime(year=1993, month=1, day=1)),
                                          lambda x: 'T4',
                                          functools.partial(get_model_by_date, 'T7','DEEP BLUE', datetime.datetime(year=1997, month=1, day=1)),
                                          functools.partial(get_model_by_date, 'T5','FAST DEEP', datetime.datetime(year=2007, month=1, day=1)),
                                          lambda x: 'T5',
                                         ],
                              },
                'TSK - TSURUMI SEIKI Co.': {'depths': [600.0, 1000.0, 2000.0],
                                            'models':  [functools.partial(get_model_by_date, 'T4','T6', datetime.datetime(year=1995, month=1, day=1)),
                                                        functools.partial(get_model_by_date, 'T5','T7', datetime.datetime(year=1979, month=1, day=1)),
                                                        lambda x: 'T5',
                                                       ]                                            
                                           }
               }

def imeta_classification(profile_data):
    if any([country1 in profile_data['country'] for country1 in TSK_countries]):
        manufacturer = MANUFACTURERS[1] # TSK
    else:
        manufacturer = MANUFACTURERS[0] # Sippican
    max_depth = profile_data['max_depth']
    profile_date = datetime.datetime(year=profile_data['year'],
                                     month=profile_data['month'],
                                     day=profile_data['day'],
                                    )
    depth_ix = get_depth_category(IMETA_VALUES[manufacturer]['depths'], max_depth)
    if depth_ix is not None:
        model_func = IMETA_VALUES[manufacturer]['models'][depth_ix]
        model = model_func(profile_date)
    else: # if the max depth is too great, we consider this not to be a valid XBT profile observation
        manufacturer = 'INVALID'
        model = 'INVALID'

    return (model, manufacturer)


def _get_arguments():
    description = ('Script to apply the intelligent metadata (imeta) algorithm to XBT profile '
                   'data to infer the instrument type from other metadata, and generate '
                   'classification metrics to compare with machine learning methods.')
    parser = argparse.ArgumentParser(description=description)
    help_msg = ('The path to the directory containing the XBT dataset in csv '
                'form, one file per year.')
    parser.add_argument('--input-path', dest='input_path', help=help_msg)
    help_msg = 'The path to where imeta classification and metric outputs will be written.'
    parser.add_argument('--output-path', dest='output_path', help=help_msg)
    help_msg = 'The start year of the range to output.'
    parser.add_argument('--start-year', dest='start_year', help=help_msg, default=None, type=int)
    help_msg = 'The end year of the range to output.'
    parser.add_argument('--end-year', dest='end_year', help=help_msg, default=None, type=int)
    
    return parser.parse_args()    

def generate_imeta():
    user_args = _get_arguments()
    print('reading data')
    xbt_full = dataexploration.xbt_dataset.XbtDataset(user_args.input_path, 
                                                       (user_args.start_year, user_args.end_year) )
    _ = xbt_full.get_ml_dataset(return_data=False)
    print('running imeta algorithm on dataset')
    imeta_classes = xbt_full.xbt_df.apply(imeta_classification, axis=1)                                                      
    imeta_feature = 'instrument_imeta'
    instrument_feature = 'instrument'
    xbt_full.xbt_df[imeta_feature] = imeta_classes
    xbt_full._feature_encoders[imeta_feature] = xbt_full._feature_encoders[instrument_feature]
    xbt_full._target_encoders[imeta_feature] = xbt_full._target_encoders[instrument_feature]
    xbt_full._output_formatters[imeta_feature] = xbt_full._output_formatters[instrument_feature]
    
    print('generating per year classification metrics for algorithm.')
    imeta_results = []
    for year in range(env_date_ranges[environment][0],env_date_ranges[environment][1]):
        y_imeta_instr = instr_encoder.transform(pandas.DataFrame(imeta_instrument[xbt_labelled.xbt_df.year == year]))
        xbt_instr1 = instr_encoder.transform(pandas.DataFrame(xbt_labelled.xbt_df[xbt_labelled.xbt_df.year == year].instrument))
        (im_pr_instr, im_rec_instr, im_f1_instr, im_sup_instr) = sklearn.metrics.precision_recall_fscore_support(xbt_instr1, y_imeta_instr,average='micro')
        imeta_results += [{'year': year,
                       'imeta_instr_recall': im_rec_instr,
                       'imeta_instr_precision': im_pr_instr,
                      }]
    
    imeta_res_df = pandas.DataFrame.from_records(imeta_results)
    
    # write metrics to imeta table
    metrics_out_path = os.path.join(user_args.output_path,
                                            'imeta_metrics.csv',
                                           )
    print(f'writing metrics to file {metrics_out_path}')
    imeta_res_df.to_csv(metrics_out_path)
    
    # write ID and imeta output to a CSV file
    classifications_out_path = os.path.join(user_args.output_path,
                                            'imeta_classifications.csv',
                                           )
    print(f'writing classifications to file {classifications_out_path}')
    xbt_full.filter_features(dataexploration.xbt_dataset.ID_FEATURES + [imeta_feature]).output_data
    print('imeta generation complete.')
    
    
                                                      
