# -*- coding: utf-8 -*-
"""
Plot the results obtained from the learning procedure
"""

import argparse
import json
import fnmatch
import os

from plot_utilities import plot_probabilities_from_dictionary
from plot_utilities import plot_scores_from_dictionary

def represent_experiment_strategy(list_of_experiments):
    """Generates a nice pdf explaining how different experiments have been set up"""
    pass 
 
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
