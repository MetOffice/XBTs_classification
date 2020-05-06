import os
import csv

import numpy

import netCDF4

# import matplotlib
# import matplotlib.pyplot
# import cartopy
# import cartopy.crs

import pandas

import multiprocessing

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

def process_xbt_file(path_nc, out_path):
    print('reading from {0}'.format(path_nc), flush=True)
    ds_nc1 = netCDF4.Dataset(path_nc)

    nc_data_dict = {}

    #TODO: split into several smaller functions for parallelisation.
    nc_data_dict['country'] = [_stringify(s1) for s1 in ds_nc1.variables['country']]
    nc_data_dict['lat'] = [float(s1) for s1 in ds_nc1.variables['lat']]
    nc_data_dict['lon'] = [float(s1) for s1 in ds_nc1.variables['lon']]
    nc_data_dict['date'] = [str(s1) for s1 in ds_nc1.variables['date']]
    #TODO: extract year, month and day
    nc_data_dict['institute'] = [_stringify(s1) for s1 in ds_nc1.variables['Institute']]
    nc_data_dict['platform'] = [_stringify(s1) for s1 in ds_nc1.variables['Platform']]
    nc_data_dict['cruise_number'] = [_stringify(s1) for s1 in ds_nc1.variables['WOD_cruise_identifier']]
    
    z_pos = numpy.insert(numpy.cumsum(ds_nc1.variables['z_row_size'][:-1]),0,0)
    nc_data_dict['instrument'] = [_stringify(s1) for s1 in ds_nc1.variables['Temperature_Instrument']]
    #TODO: add model and manufacturer
    
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
    print('writing to {0}'.format(out_path), flush=True)
    df_nc1.to_csv(out_path)
    return df_nc1

def process_year(year1, nc_dir, out_dir):
    print('processing year {0}'.format(year1), flush=True)
    fname1 = 'iquod_xbt_{year}.nc'.format(year=year1)
    path_nc = os.path.join(nc_dir, fname1)

    out_path_template = os.path.join(out_dir, 'xbt_{year}.csv')
    out_path = out_path_template.format(year=year1)
    df_nc1 = process_xbt_file(path_nc, out_path)

def main():
    base_dir = os.environ['BASE_DIR']
    nc_dir = os.path.join(base_dir, os.environ['NC_DIR_NAME'])
    out_dir = os.path.join(base_dir, os.environ['OUTPUT_DIR_NAME'])
    start_year = int(os.environ['START_YEAR'])
    end_year = int(os.environ['END_YEAR'])
    try:
        pool_size = int(os.environ['NTASKS'])
    except KeyError as ke1:
        pool_size = 1
    print('Running {0} XBT tasks'.format(pool_size))
    pool1 = multiprocessing.Pool(pool_size)
    arg_list = [(y1,nc_dir,out_dir) for y1 in range(start_year,end_year+1)]
    pool1.starmap(process_year, arg_list)
    pool1.close()
    pool1.join()

    # process_year(1982, nc_dir, out_dir, True)

if __name__ == '__main__':
    main()
