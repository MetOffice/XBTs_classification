# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:09:38 2017

@author: Francesco Capponi

Run the k-nearest-neighbor classification for a given year
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

RELEVANT_FEATURES = ['lat','lon','max_depth']
TARGET = 'instrument'

    
def optimize(X_train,y,max_nneighbors,n_folds):
    """Apply k-fold cross-validation to tune the optimal number of neighbors"""
    
    scores = []
    times=[]
    data_to_plot = []
   
    for k in range(1,max_nneighbors):
        t1 = time.time()
        classifier = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
        optimizer = cross_val_score(classifier, X_train, y, cv=n_folds, n_jobs=-1)
        t_train=time.time() - t1
        times.append(t_train)
        scores.append([k,np.mean(optimizer),np.std(optimizer)])
        data_to_plot.append(list(optimizer))

    optimal_nneighbors=max(scores,key=lambda x: x[1])
    return optimal_nneighbors, scores, times, data_to_plot

def plot_output(optimal_nneighbors, scores, times, data_to_plot, max_nneighbors,year):
    """Create plots describing cross-validation procedure and computation times"""    
    
    print("Plotting information about hyperparameter tuning for year "+year)
    
    plot_dir=os.getcwd()+'/knn_tuning'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)    
    
    nneighbors=range(1,30)
    fig = plt.figure()
    plt.plot(nneighbors,np.array(data_to_plot), ':')
    plt.plot(nneighbors, [row[1] for row in  scores], 'k',label='Average across the folds', linewidth=2)
    plt.axvline(optimal_nneighbors[0], linestyle='--', color='k', label='tree depth: CV estimate')
    plt.legend()
    plt.xlabel('N neighbors')
    plt.ylabel('Accuracy score')
    plt.title('Mean accuracy score on each fold')
    plt.axis('tight')
    plt.close(fig)
    
    fig.savefig(plot_dir+"/K-fold_cross_validation"+year+".pdf")

def classify(train_name,test_name,year,log_tuning):
    """Run classification for a single year"""
    
    # Loading the data, dropping nan
    train, test = pd.read_csv(train_name), pd.read_csv(test_name)
    train, test = train.dropna(), test.dropna()
    
    # Selecting input and output features
    X_train,X_test = train[RELEVANT_FEATURES].values,test[RELEVANT_FEATURES].values
    y_train,y_test = train[TARGET],test[TARGET]
   
    # Rescaling
    X_train,X_test = rescale(X_train,X_test)

    # Optimizing the hyperparameter    
    max_nneighbors = 30
    optimal_nneighbors, scores, times, data_to_plot = optimize(X_train,y_train,max_nneighbors,5)
    
    # Reporting information about tuning procedure, if required
    if log_tuning:
        plot_output(optimal_nneighbors, scores, times, data_to_plot, max_nneighbors, year)
    
    # Prediction
    classifier = KNeighborsClassifier(n_neighbors = optimal_nneighbors[0], weights = 'distance') 
    classifier.fit(X_train,y_train)
    prediction = classifier.predict(X_test)
    return({'accuracy_score':accuracy_score(y_test,prediction),'N_neighbors':optimal_nneighbors[0]})
    
def main():
    """Run analysis for a single year"""
    
    parser = argparse.ArgumentParser(description='Run the k-nearest-neighbor classification for a given year.\n')
    parser.add_argument('--path',default='./data/',help='input train and test files location')
    parser.add_argument('-train_prefix',default='train_',help='training set name prefix')
    parser.add_argument('-test_prefix',default='test_',help='test set name prefix')
    parser.add_argument('-log_tuning',default=0,help='enable production of plots for checking the tuning procedure')
    parser.add_argument('year',help='year')
    
    args = parser.parse_args()
    train_name=args.path+args.train_prefix+args.year+'.csv'
    test_name=args.path+args.test_prefix+args.year+'.csv'
     
    result = classify(train_name,test_name,args.year,args.log_tuning)
    print(result)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()