import os
import csv
import re
import argparse
import multiprocessing
import datetime
import time
import tempfile

import numpy
import netCDF4
import pandas


IQUOD_TEMPLATE = 'iquod_xbt_{year}.nc'
XBT_OUT_TEMPLATE = 'xbt_{year}.csv'

DEFAULT_TEMP_DIR = '/scratch'

DEFAULT_PREPROC_TASKS = 4

def _stringify(cell):
    s = ''.join(cell.data[numpy.invert(cell.mask)].astype(numpy.str))
    if s == '':
        s = '0'
    return s


def get_csv_data(path_csv):
    data_lines_csv1 = []
    with open(path_csv,'r') as file_csv1:
        reader1 = csv.reader(file_csv1)
        for csv_line1 in reader1:
            data_lines_csv1 += [csv_line1]
            ix1 += 1
    return data_lines_csv1

INSTRUMENT_REGEX_STRING = 'XBT[:][\s](?P<model>[\w\s;:-]+)([\s]*)([(](?P<manufacturer>[\w\s.:;-]+)[)])?'
REGEX_MANUFACTURER_GROUP = 'manufacturer'
REGEX_MODEL_GROUP = 'model'
UNKNOWN_STR = 'UNKNOWN'

def get_model(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        type_str = matches.group(REGEX_MODEL_GROUP)
    except AttributeError as e1:
        type_str = UNKNOWN_STR
    return str(type_str).strip(' ')


def get_manufacturer(instr_str):
    try:
        matches = re.search(INSTRUMENT_REGEX_STRING, instr_str)
        brand_str = matches.group(REGEX_MANUFACTURER_GROUP)
    except AttributeError as e1:
        brand_str = UNKNOWN_STR
    return str(brand_str).strip(' ')

def get_year(dt_str):
    try:
        dt1 = datetime.datetime.strptime(dt_str,'%Y%m%d')
        year = dt1.year
        month = dt1.month
        day = dt1.day
    except ValueError:
        year = 0
        month=0
        day = 0
    return dt_str, year, month, day


def process_xbt_file(path_nc, out_path=None):
    print('reading from {0}'.format(path_nc), flush=True)
    ds_nc1 = netCDF4.Dataset(path_nc)

    nc_data_dict = {}

    #TODO: split into several smaller functions for parallelisation.
    nc_data_dict['country'] = [_stringify(s1) for s1 in ds_nc1.variables['country']]
    nc_data_dict['lat'] = [float(s1) for s1 in ds_nc1.variables['lat']]
    nc_data_dict['lon'] = [float(s1) for s1 in ds_nc1.variables['lon']]
    
    date_info_tuples = [get_year(str(s1)) for s1 in ds_nc1.variables['date']]
    nc_data_dict['date'] = [t1[0] for t1 in date_info_tuples]
    nc_data_dict['year'] = [t1[1] for t1 in date_info_tuples]
    nc_data_dict['month'] = [t1[2] for t1 in date_info_tuples]
    nc_data_dict['day'] = [t1[3] for t1 in date_info_tuples]
    
    nc_data_dict['institute'] = [_stringify(s1) for s1 in ds_nc1.variables['Institute']]
    nc_data_dict['platform'] = [_stringify(s1) for s1 in ds_nc1.variables['Platform']]
    nc_data_dict['cruise_number'] = [_stringify(s1) for s1 in ds_nc1.variables['WOD_cruise_identifier']]
    
    z_pos = numpy.insert(numpy.cumsum(ds_nc1.variables['z_row_size'][:-1]),0,0)
    nc_data_dict['instrument'] = [_stringify(s1) for s1 in ds_nc1.variables['Temperature_Instrument']]
    nc_data_dict['model'] = [get_model(s1) for s1 in nc_data_dict['instrument']]
    nc_data_dict['manufacturer'] = [get_manufacturer(s1) for s1 in nc_data_dict['instrument']]
    
    nc_data_dict['temperature_profile'] = [ds_nc1.variables['Temperature'][z_pos1:z_pos1+zl1].data.tolist() for (z_pos1,zl1) in zip(z_pos,ds_nc1.variables['Temperature_row_size'])]
    nc_data_dict['temperature_quality_flag'] = [ds_nc1.variables['Temperature_IQUODflag'][z_pos1:z_pos1+zl1] for (z_pos1,zl1) in zip(z_pos,ds_nc1.variables['Temperature_row_size'])]
    
    nc_data_dict['depth_profile'] = [ds_nc1.variables['z'][z_pos1:z_pos1+zl1].data.tolist() for (z_pos1,zl1) in zip(z_pos,ds_nc1.variables['z_row_size'])]
    nc_data_dict['max_depth'] = [dp1[-1] for dp1 in nc_data_dict['depth_profile']]
    nc_data_dict['depth_quality_flag'] = [ds_nc1.variables['z_IQUODflag'][z_pos1:z_pos1+zl1] for (z_pos1,zl1) in zip(z_pos,ds_nc1.variables['z_row_size'])]
    
    lut1 = {-9: 0, 1: 1}
    try:
        nc_data_dict['imeta_applied'] = [lut1[i1] for i1 in ds_nc1.variables['Temperature_Instrument_intelligentmetadata'][:].data]
    except KeyError:
        # if no imeta has been applied, the Temperature_Instrument_intelligentmetadata variable may not be present
        # in which case, create an array which is all zeroes, to represent no imeta applied. 
        nc_data_dict['imeta_applied'] = [0] * ds_nc1.variables['Temperature_Instrument'].shape[0]
    
    # add depth equation
    nc_data_dict['id'] = [int(s1) for s1 in ds_nc1.variables['wod_unique_cast']]

    df_nc1 = pandas.DataFrame(nc_data_dict)
    if out_path is not None:
        print('writing to {0}'.format(out_path), flush=True)
        df_nc1.to_csv(out_path)
    return df_nc1

def process_year(year1, nc_dir, out_dir):
    print('processing year {0}'.format(year1), flush=True)
    fname1 = IQUOD_TEMPLATE.format(year=year1)
    path_nc = os.path.join(nc_dir, fname1)

    out_path_template = os.path.join(out_dir, XBT_OUT_TEMPLATE)
    out_path = out_path_template.format(year=year1)
    df_nc1 = process_xbt_file(path_nc, out_path)
    
def process_file(nc_path, out_dir, temp_dir, xbt_ix):
    print(f'processing file {nc_path}')
    temp_out_path = os.path.join(temp_dir, f'xbt_temp_{xbt_ix}.csv')
    _ = process_xbt_file(nc_path, temp_out_path) 
    return temp_out_path
    
def get_input_file_list(nc_dir, regex_str):
    return [os.path.join(nc_dir,f1) for f1 in os.listdir(nc_dir) if re.search(regex_str, f1) ]


def process_args():
    description = ('Script to preprocess data downloaded from the WOD, only '
                   'XBT relevant metadata, do basic preprocessing, and split '
                   'into CSV files per year.')
    parser = argparse.ArgumentParser(description=description)
    help_msg = 'The path to the directory containing the WOD netCDF4 files with XBT profiles.'
    parser.add_argument('--input-path', dest='input_path', help=help_msg,  required=True)
    help_msg = 'The path to the directory for outputting per year XBT CSV files.'
    parser.add_argument('--output-path', dest='output_path', help=help_msg, required=True)
    help_msg = ('The directory to write out the intermediate CSV files after '
                'preprocessing before they are join and split by year.')
    parser.add_argument('--temp-path', dest='temp_path', help=help_msg, default=DEFAULT_TEMP_DIR)
    help_msg = 'The start year of the range to output.'
    parser.add_argument('--start-year', dest='start_year', help=help_msg, default=None, type=int)
    help_msg = 'The end year of the range to output.'
    parser.add_argument('--end-year', dest='end_year', help=help_msg, default=None, type=int)
    help_msg = 'The number of parallel tasks to use.'
    parser.add_argument('--num-tasks', dest='num_tasks', help=help_msg, default=1, type=int)
    help_msg = 'The characters at the start of the input file names.'
    parser.add_argument('--prefix', help=help_msg, required=True)
    help_msg = 'The characters at the end of the input file names.'
    parser.add_argument('--suffix', help=help_msg, required=True)
    return parser.parse_args()    

def do_wod_extract(nc_dir, out_dir, fname_prefix, fname_suffix, temp_dir, start_year=None, end_year=None, pool_size=DEFAULT_PREPROC_TASKS):
    start1 = time.time()
    pattern1 = fname_prefix + '([\w\d\.]+)' + fname_suffix        
    nc_path_list = get_input_file_list(nc_dir, pattern1)
    print('found files to process:\n' + '\n'.join(nc_path_list) + '\n')
    print('Running {0} XBT tasks'.format(pool_size))
    pool1 = multiprocessing.Pool(pool_size)
    arg_list = [(nc_path, out_dir, temp_dir, ix1) for ix1, nc_path in enumerate(nc_path_list)]
    xbt_tempfile_list = pool1.starmap(process_file, arg_list)
    pool1.close()
    pool1.join()

    print('reading in interediate files.')    
    xbt_full = pandas.concat([pandas.read_csv(path1) for path1 in xbt_tempfile_list])
    xbt_full = xbt_full[xbt_full.year > 0] # some profiles have year set to 0, ignore these profiles.
    if start_year is None:
        start_year = min(xbt_full.year.unique())
    if end_year is None:
        end_year = max(xbt_full.year.unique())
        
    print('writing out per year files.')
    for year1 in range(start_year, end_year+1):
        print(f'writing year {year1}')
        year_out_path = os.path.join(out_dir, XBT_OUT_TEMPLATE.format(year=year1))
        if sum(xbt_full.year == year1) > 0:
            xbt_full[xbt_full.year == year1].to_csv(year_out_path)

    end1 = time.time()
    duration1 = end1-start1
    print(f'preprocessing duration: {duration1:.3f} seconds')

            
def wod_extract():
    
    user_args = process_args()
    nc_dir = user_args.input_path
    out_dir = user_args.output_path
    start_year = user_args.start_year
    end_year = user_args.end_year
    fname_prefix =  user_args.prefix
    fname_suffix =  user_args.suffix
    pool_size = user_args.num_tasks
    temp_path = user_args.temp_path
    # create a directory for intermediate files, which is automatically cleaned up.
    with tempfile.TemporaryDirectory(dir=temp_path) as temp_dir:
        do_wod_extract(nc_dir=nc_dir, 
                       out_dir=out_dir, 
                       temp_dir=temp_dir,
                       start_year=start_year, 
                       end_year=end_year, 
                       fname_prefix=fname_prefix, 
                       fname_suffix=fname_suffix, 
                       pool_size=pool_size,
                      )
            
        

if __name__ == '__main__':
    main()
