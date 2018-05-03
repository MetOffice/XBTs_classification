"""Data exploration module"""

import numpy
import os
import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_unknown_brand_probes(inpath, outpath, logging=True):
    """Perform year-by-year exploration of the dataset, by reading the list of available files, produce latitude-vs-longitude plots for XBT data instances with known type and unknown brand"""

    list_of_files = os.listdir(inpath)
    list_of_files.sort()
    
    message = 'We have {0} input files'.format(len(list_of_files))
    if logging:
        print(message)

    input_variable = 'instrument'
    target_variables = ['cruise_number', 'platform']
    latitude_label, longitude_label = 'lat', 'lon'
    relevant_columns = [latitude_label, longitude_label]+target_variables

    for index, data_file in enumerate(list_of_files):

        name = os.path.join(inpath, data_file)
        dataset = pandas.DataFrame.from_csv(name)
        
        year = data_file.replace('xbt_','')
        year = year.replace('.csv','')
        
        unique_list_of_probes = dataset[input_variable].unique()
        unique_unknown_brand = [item for item in unique_list_of_probes if 'UNKNOWN BRAND' in item]

        if len(unique_unknown_brand):
            if logging:
                print(year + ":these are the following XBTs data instances with known type and unknown brand")
                print(unique_unknown_brand)
        
            for target_variable in target_variables:
                out_dir = os.path.join(outpath,target_variable+'_'+year)
                        
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                display_unknown_brand_data_instances(dataset, unique_unknown_brand, 
                                                     latitude_label, longitude_label, 
                                                     relevant_columns,
                                                     input_variable, target_variable,
                                                     out_dir)

def display_unknown_brand_data_instances(data_frame, unique_unknown_brand, latitude_label, longitude_label, relevant_columns, input_variable, target_variable, outpath):
    """Represent XBTs data instances as latitude-vs-longitude plots, labeling data points with the given target variable"""
    
    color_block = 10

    for brand in unique_unknown_brand:
        
        brandname = brand.replace('XBT: ','')
        brandname = brandname.replace(' ','_')
        brand_outpath=os.path.join(outpath, brandname)

        sliced_frame = data_frame[data_frame[input_variable]==brand][relevant_columns]
        target_values = numpy.unique(sliced_frame[target_variable].values)

        with PdfPages(brand_outpath+'.pdf') as pdf:
            
            # the number of unique target lables can be quite large, causing repetitions of color codes 
            counter_range = target_values.size//color_block
            if not counter_range:
                counter_range=1
                
            for counter in range(counter_range):
                plt.figure()
                
                for target_value in target_values[counter*color_block:(counter+1)*color_block]:
                    lat = sliced_frame[sliced_frame[target_variable]==target_value][latitude_label].values
                    lon = sliced_frame[sliced_frame[target_variable]==target_value][longitude_label].values
                    plt.plot(lon, lat, '.', label=target_value)
                                            
                    plt.xlabel(longitude_label)
                    plt.ylabel(latitude_label)
                    plt.title(brand)
                    
                for target_value in target_values[(counter+1)*color_block:-1]:
                    lat = sliced_frame[sliced_frame[target_variable]==target_value][latitude_label]
                    lon = sliced_frame[sliced_frame[target_variable]==target_value][longitude_label]
                    plt.plot(lon, lat, '.', label=target_value)
                    
                if target_variable=='cruise_number':
                    plt.legend(loc='center right', bbox_to_anchor=(1.4, .5))
                else:
                    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5))

                plt.xlabel(longitude_label)
                plt.ylabel(latitude_label)
                plt.title(brand)
                    
                counter+=1
                pdf.savefig(bbox_inches="tight")
                plt.close()
