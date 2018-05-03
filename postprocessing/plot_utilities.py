"""Collection of plotting utilities"""

import matplotlib.pyplot as plt
import numpy
import os
NAMES = ['accuracy', 'recall']

def plot_scores_from_dictionary(dict_of_dictionaries, plot_dir, prefix):
    """Plot the values of accuracy, recall and precision along the years, for all the experiments"""

    # loop over all the metrics
    for name in NAMES:        
        metric_dictionary = {}
        year = {}
        for key, item in dict_of_dictionaries.iteritems():
            years = []
            metrics = []
            for sub_item in item:
                years.append(int(sub_item['year']))
                metrics.append(sub_item[name])
            metric_dictionary[key] = numpy.array(metrics)
            year[key] = numpy.array(years)

        #plotting metric
        keys = dict_of_dictionaries.keys()
        fig = plt.figure()
        
        for key in keys:
            sorted_indices = numpy.argsort(year[key])
            plt.plot(year[key][sorted_indices], metric_dictionary[key][sorted_indices])
        plt.xlabel('years')
        plt.ylabel(name)
        plt.title(prefix+' '+name)
        plt.axis('tight')
        plt.legend(keys)
        outpath = os.path.join(plot_dir, prefix +'_' + name +'_scores.pdf')
        fig.savefig(outpath)
        plt.close(fig)
    
def plot_probabilities_from_dictionary(dictionary, plot_dir, year, prefix, directory):
    """Plot the predicted probabilities for a given year, for all the experiments"""
    
    probabilities = numpy.array(dictionary['probabilities'])
    class_mapping = dictionary['class_mapping']
    
    # the x locations for the groups
    ind = numpy.arange(len(probabilities))  
    # the width of the bars
    width = 0.35                    
                                             
    ordered_numerical_keys = sorted([int(key) for key in class_mapping.keys()])
    labels = list([class_mapping[str(key)] for key in ordered_numerical_keys])
    
    fig, ax = plt.subplots()
    ax.bar(ind, probabilities, width, color='b')
    
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels, rotation = 30, size = 5, rotation_mode = 'anchor')
    plt.xlabel('class_code')
    plt.ylabel('class_probability')
    plt.title(year + ', ' + directory + ': ' + prefix + ' class probabilities')
    plt.axis('tight')
    outpath = os.path.join(plot_dir, year + '_' + directory + '_' + prefix +'_predicted_probabilities.pdf')
    fig.savefig(outpath)
    plt.close(fig) 
