"""
Runs several machine learning experiments for a given year
"""

import argparse
import fnmatch
import os
from classification import ClassificationExperiment
from classification import run_experiment

def main():
    parser = argparse.ArgumentParser(description='Runs several machine learning experiments for a given year of analysis')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='outputpath for results')
    parser.add_argument('--year', default='',help='year')
    parser.add_argument('train',help='training set name')
    parser.add_argument('test',help='test set name')
    parser.add_argument('json_descriptors_folder',help='path to the folder containing json descriptors')
    
    args = parser.parse_args()
    print(args.json_descriptors_folder)
    list_of_files = os.listdir(args.json_descriptors_folder)
    print(list_of_files)
    list_of_json_files = []
    for file in list_of_files:
        if fnmatch.fnmatch(file, '*.json'):
            #check all the json descriptors have been correctly written
            ClassificationExperiment(None,None,file).read_json_file()
            list_of_json_files.append(file)
            
    for descriptor in list_of_json_files:
        run_experiment(args.path, args.outpath, args.year, args.train, args.test, descriptor)

if __name__ == "__main__":
    # execute only if run as a script
    main()