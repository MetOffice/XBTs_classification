# -*- coding: utf-8 -*-
"""
Plot the results of the learning phase
"""

import argparse
import json
import fnmatch
import matplotlib.pyplot as plt
import numpy
import os

def plot_scores_from_dictionary(dict_of_dictionaries, plot_dir, prefix):
    """Plot the values of accuracy, recall and precision along the years, for all the experiments"""
        
    accuracy = {}
    recall = {}
    year = {}
    for key, item in dict_of_dictionaries.iteritems():
        years = []
        recalls = []
        accuracies = []
        for sub_item in item:
            years.append(int(sub_item['year']))
            recalls.append(sub_item['recall'])
            accuracies.append(sub_item['accuracy'])
        accuracy[key] = numpy.array(accuracies)
        recall[key] = numpy.array(recalls)
        year[key] = numpy.array(years)

    #plotting accuracy
    keys = dict_of_dictionaries.keys()
    fig = plt.figure()
        
    for key in keys:
        sorted_indices = numpy.argsort(year[key])
        plt.plot(year[key][sorted_indices], accuracy[key][sorted_indices])
    plt.xlabel('years')
    plt.ylabel('accuracy')
    plt.title(prefix+' accuracy score')
    plt.axis('tight')
    plt.legend(keys)
    outpath = os.path.join(plot_dir, prefix +'_accuracy_scores.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    
    fig = plt.figure()
       
    for key in keys:
        sorted_indices = numpy.argsort(year[key])
        plt.plot(year[key][sorted_indices], recall[key][sorted_indices])
        
    plt.xlabel('years')
    plt.ylabel('recall')
    plt.title(prefix+' recall score')
    plt.axis('tight')
    plt.legend(keys)
    outpath = os.path.join(plot_dir, prefix +'_recall_scores.pdf')
    fig.savefig(outpath)
    plt.close(fig)
 
def plot_probabilities_from_dictionary(dictionary, plot_dir, year, prefix, directory):
    """Plot the predicted probabilities for a given year, for all the experiments"""
    
    probabilities = numpy.array(dictionary['probabilities'])
    class_mapping = dictionary['class_mapping']
        
    #print(probabilities)
    #print(class_mapping.values())
    ind = numpy.arange(len(probabilities))  # the x locations for the groups
    width = 0.35       # the width of the bars
 
    fig, ax = plt.subplots()
    ax.bar(ind, probabilities, width, color='b')
    #ax.plot()

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(class_mapping.values(), rotation = 30, size = 5, rotation_mode = 'anchor')
    plt.xlabel('class_code')
    plt.ylabel('class_probability')
    plt.title(prefix+' class probabilities for year '+year)
    plt.axis('tight')
    outpath = os.path.join(plot_dir, year + '_' + directory + '_' + prefix +'_predicted_probabilities.pdf')
    fig.savefig(outpath)
    plt.close(fig) 

def main():
    """Plot results for all the experiments that have been run"""
    
    parser = argparse.ArgumentParser(description='Plot results for all the experiments that have been run\n')
    parser.add_argument('path',help='main directory containing experiments results')
        
    args = parser.parse_args()
      
    plot_dir = os.path.join(args.path, 'results_plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)    

    list_of_dirs = os.listdir(args.path)

    for prefix in ['type','type_and_manifacturer']:
        dict_of_dictionaries = {}

        for directory in list_of_dirs:
            basepath = os.path.join(args.path,directory)
            json_results = os.listdir(basepath)
  
            temp_list = []
            for json_file in json_results:
                if fnmatch.fnmatch(json_file, '*'+prefix+'_prediction*'):
                    year = os.path.basename(json_file).split('_')[0]
                    filename  = os.path.join(basepath, json_file)
                    
                    with open(filename) as json_data:
                        dictionary = json.load(json_data)
                        temp_list.append({'year':year, 'accuracy': dictionary['accuracy'], 'recall':dictionary['recall']})
                        json_data.close()
                        
                        # plotting predicted probabilities for a single year
                        plot_probabilities_from_dictionary(dictionary, plot_dir, year, prefix, directory)
            
            if directory != os.path.basename(plot_dir):
                dict_of_dictionaries[directory] = temp_list

        # comparison of scores among experiments through all the years
        plot_scores_from_dictionary(dict_of_dictionaries, plot_dir, prefix)
              
if __name__ == "__main__":
    # execute only if run as a script
    main()
