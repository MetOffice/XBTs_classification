#!/usr/bin/env python
import bokeh.plotting
import bokeh.layouts
import bokeh.models.widgets
import pandas
import json
import argparse
import cartopy.feature
import re
import os

XBT_YEAR_WIDGET = 'xbt_year'
CRUISE_WIDGET = 'cruise_id'

def on_click_load_xbt():
    print('on click load')
    bokeh_doc = bokeh.io.curdoc()
    year_str = bokeh_doc.widgets[XBT_YEAR_WIDGET].value
    selected_year = int(year_str)
    if bokeh_doc.current_data_key == selected_year:
        print(f'No change to selected year {selected_year}')
        return

    bokeh_doc.current_data_key = selected_year
    input_fname = bokeh_doc.input_data[bokeh_doc.current_data_key]
    input_path = os.path.join(bokeh_doc.input_data_dir,
                              input_fname)
    print(f'loading data for {selected_year} from {input_path}')
    xbt_df = pandas.read_csv(input_path)

    print('Updating data frame')
    bokeh_doc.xbt_df = xbt_df
    bokeh_doc.data_source.data = bokeh_doc.data_source.from_df(xbt_df)
    print('Updating cruise ID list')
    cruise_id_list = _get_cruise_id_list(xbt_df)
    bokeh_doc.cruise_id_list = cruise_id_list
    bokeh_doc.widgets[CRUISE_WIDGET].options = cruise_id_list
    bokeh_doc.widgets[CRUISE_WIDGET].value = 'all'
    print('finished data load update')

def on_select_cruise(attrname, old_val, new_val):
    print('on_select_cruise')
    bokeh_doc = bokeh.io.curdoc()
    if new_val == bokeh_doc.selected_cruise:
        print('No change in selected cruise {0}'.format(bokeh_doc.selected_cruise))
        return

    if bokeh_doc.xbt_df is not None:
       selected_cruise = new_val
       print(f'selecting items from cruise {selected_cruise}')
       xbt_df = bokeh_doc.xbt_df

       if selected_cruise =='all':
           bokeh_doc.data_source.data = bokeh_doc.data_source.from_df(xbt_df)
       else:
           bokeh_doc.data_source.data = bokeh_doc.data_source.from_df(xbt_df[xbt_df['cruise_number'].str.match(selected_cruise)])
    else:
        print('No XBT DF present, taking no actions')

def _get_cruise_id_list(xbt_year_csv):
    cruise_id_list = list(xbt_year_csv.cruise_number.unique())
    cruise_id_list += ['all']
    return cruise_id_list

XBT_FILE_REX_STR = r'xbt_(?P<year>\d{4}).csv'


def launch_bokeh_explorer(input_data_dir):
    bokeh_doc = bokeh.io.curdoc()
    bokeh_doc.input_data_dir = input_data_dir
    # input_files = os.listdir
    pattern_xbt_file = re.compile(XBT_FILE_REX_STR)
    input_data = {
        int(pattern_xbt_file.search(f1).groupdict()['year']): f1
        for f1 in os.listdir(input_data_dir)
        if 'csv' in f1
    }
    bokeh_doc.input_data = input_data
    bokeh_doc.current_data_key = 1975

    xbt_columns = ['country', 'lat', 'lon', 'date', 'institute', 'platform',
               'cruise_number', 'instrument', ]
    bokeh_doc.xbt_columns = xbt_columns
    bokeh_doc.xbt_df = None

    bokeh_doc.mip_tables = None

    print(f'reading data from {input_data}')
    input_path = os.path.join(input_data_dir,
                              input_data[bokeh_doc.current_data_key])
    xbt_year_csv = pandas.read_csv(input_path)
    bokeh_doc.xbt_df = xbt_year_csv

    cruise_id_list = _get_cruise_id_list(xbt_year_csv)
    bokeh_doc.cruise_id_list = cruise_id_list
    bokeh_doc.selected_cruise = 'all'



    file_path_txt_input = bokeh.models.widgets.inputs.TextInput(
        value=input_data_dir)
    data_source = bokeh.models.ColumnDataSource(xbt_year_csv)
    bokeh_columns = [bokeh.models.TableColumn(field=col_name, title=col_name) for col_name in xbt_columns]
    variable_table = bokeh.models.widgets.DataTable(source=data_source, columns=bokeh_columns)
    # sample_locations_figure = bokeh.plotting.figure(plot_height=600,
    #                                                 plot_width=700, title="",
    #                                                 toolbar_location=None,
    #                                                 sizing_mode="scale_both")
    # sample_locations_figure.cross(x='lon',
    #                               y='lat',
    #                               source=data_source,
    #                               size=3,
    #                               color='blue',
    #                               )

    # for val in cartopy.feature.COASTLINE.geometries():
    #     coast_lons, coast_lats = val.xy
    #
    #     sample_locations_figure.multi_line(xs=[list(coast_lons)],
    #                                        ys=[list(coast_lats)],
    #                                        color='red')


    bokeh_doc.data_source = data_source

    xbt_year_select = bokeh.models.widgets.Select(name='Year select')
    xbt_year_select.options = [str(i1) for i1 in bokeh_doc.input_data.keys()]
    xbt_year_select.value = str(bokeh_doc.current_data_key)
    load_xbt_button = bokeh.models.widgets.buttons.Button(label='Get XBT data')
    cruise_select = bokeh.models.widgets.Select(name='Cruise ID')
    cruise_select.on_change('value', on_select_cruise)
    cruise_select.options = cruise_id_list
    cruise_select.value = 'all'
    load_xbt_button.on_click(on_click_load_xbt)
    main_layout = bokeh.layouts.column(
        file_path_txt_input,
        xbt_year_select,
        load_xbt_button,
        cruise_select,
        variable_table,
        # sample_locations_figure,
    )
    bokeh_doc.widgets = {CRUISE_WIDGET: cruise_select,
                        'load_button': load_xbt_button,
                         XBT_YEAR_WIDGET: xbt_year_select}
    bokeh_doc.add_root(main_layout)
    bokeh_doc.title='XBT data browser'

def get_args():
    parser1 = argparse.ArgumentParser()
    help_msg = 'The data file with XBT input data in CSV format to be ' \
               'explored.'
    parser1.add_argument('input_data_dir',
                         type=str,
                         default='',
                         help=help_msg)
    cmd_args = parser1.parse_args()
    return cmd_args

def main():
    cmd_args = get_args()
    launch_bokeh_explorer(**cmd_args.__dict__)


if 'bk_script' in __name__  or \
    'main' in __name__:
    main()

#TODO list
# get plotting to work, without disconnecting server
# get column widths to display correctly
# be able to select by other criteria
# display multiple years?
