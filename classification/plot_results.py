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
            years.append(sub_item['year'])
            recalls.append(sub_item['recall'])
            accuracies.append(sub_item['accuracy'])
        accuracy[key] = numpy.array(accuracies)
        recall[key] = numpy.array(recalls)
        year[key] = numpy.array(years)

    #plotting accuracy
    keys = dict_of_dictionaries.keys()
    fig = plt.figure()
        
    for key in keys:
        plt.plot(year[key], accuracy[key])
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
        plt.plot(year[key], recall[key])
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
              
""" 

{"input_features": {"features_to_get_labeled": ["instrument_type", "instrument_type_and_manifacturer"], "features_to_get_dummy": ["platform"], "useless_features": ["date", "depth_profile", "institute", "country"], "features_to_rescale": ["max_depth", "lat", "lon", "platform"]}, "probabilities": [0.0, 0.7391283520382338, 0.010069481486189488, 0.009366140113847201, 0.016169654586376012, 0.004610484715706066, 0.020031721976796996, 0.2006241650828502], "recall": 0.8568414218471028, "applied_operations": ["map_zero_platform_to NaN", "negative_max_depth_to_NaN", "generate_target_outputs"], "best_hyperparameters": {"n_neighbors": 8, "n_jobs": -1, "weights": "distance"}, "class_mapping": {"0": "AXBT 536", "1": "DEEP BLUE", "2": "FAST DEEP", "3": "T10", "4": "T4", "5": "T5", "6": "T6", "7": "T7"}, "rescale_all": 1, "accuracy": 0.8568414218471028}
   
    fig = plt.figure()
    plt.plot(result_frame.index.astype(int),result_frame['accuracy_score'].values, ':',label='accuracy_score')
    plt.legend()
    plt.xticks(np.arange(int(args.startyear),int(args.endyear), 1))
    plt.xlabel('year')
    plt.ylabel('Accuracy score')
    plt.title('Accuracy score from'+args.startyear+' to '+args.endyear)
    plt.axis('tight')
    fig.savefig(plot_dir+"/classification_score.pdf")
    plt.close(fig)    
    
    fig = plt.figure()
    plt.plot(result_frame.index.astype(int),result_frame['N_neighbors'].values, ':',label='accuracy_score')
    plt.legend()
    plt.xticks(np.arange(int(args.startyear),int(args.endyear), 1))
    plt.xlabel('year')
    plt.ylabel('N neighbors')
    plt.title('N neighbors evolution from'+args.startyear+' to '+args.endyear)
    plt.axis('tight')
    fig.savefig(plot_dir+"/n_neighbors_evolution.pdf")
       
    out_file='/results.csv'
    result_frame.to_csv(plot_dir+out_file, index_label='year')
    t_classification=time.time() - t1
    print('time elapsed = ',t_classification,' seconds')
    plt.close(fig)
"""
if __name__ == "__main__":
    # execute only if run as a script
    main()
