# -*- coding: utf-8 -*-
"""
Plot the results of the learning phase
"""

def main():
    """Run analysis for all years"""
    
    t1 = time.time()
    parser = argparse.ArgumentParser(description='Run the k-nearest-neighbor classification for all the years.\n')
    parser.add_argument('--path',default='./data/',help='input train and test files location')
    parser.add_argument('-train_prefix',default='train_',help='training set name prefix')
    parser.add_argument('-test_prefix',default='test_',help='test set name prefix')
    parser.add_argument('-log_tuning',default=0,help='enable production of plots for checking the tuning procedure')
    parser.add_argument('-startyear',default=str(1970),help='start year')    
    parser.add_argument('-endyear',default=str(2017),help='end year')
        
    args = parser.parse_args()
     
    result = {}
    for year in range(int(args.startyear),int(args.endyear)):
        train_name=args.path+args.train_prefix+str(year)+'.csv'
        test_name=args.path+args.test_prefix+str(year)+'.csv'
        result[str(year)] = KNN.classify(train_name,test_name,str(year),args.log_tuning)
    
    result_frame=pd.DataFrame.from_dict(result,orient='index')
    
    #Plotting classification result
    
    plot_dir=os.getcwd()+'/knn_result'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)    
    
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
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
