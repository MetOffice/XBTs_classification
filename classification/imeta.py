"""
This algorithm for detemining the model and manufacturer is taken from the following paper by Palmer et. al.
https://journals.ametsoc.org/doi/full/10.1175/JTECH-D-17-0129.1
"""
import os
import datetime
import functools

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
IMETA_VALUES = { 'SIPPICAN' : { 'depths': [360.0,600.0,1000.0,1350.0,2300.0],
                               'models': [functools.partial(get_model_by_date, 'T4','T10', datetime.datetime(year=1993, month=1, day=1)),
                                          lambda x: 'T4',
                                          functools.partial(get_model_by_date, 'T7','DEEP BLUE', datetime.datetime(year=1997, month=1, day=1)),
                                          functools.partial(get_model_by_date, 'T5','FAST DEEP', datetime.datetime(year=2007, month=1, day=1)),
                                          lambda x: 'T5',
                                         ],
                              },
                'TSK - TSURUMI SEIKI Co.': {'depths': [600.0, 1000.0, 2300.0],
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

