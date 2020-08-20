import os
import pathlib

import pandas

class WODInstrumentCodes():
    WOD_INSTRUMENT_CODES_FILENAME = 'v_5_instrument.csv'
    XBT_PREFIX = 'XBT'
    def __init__(self):
        code_file_dir = pathlib.Path(__file__).absolute().parent
        self.df_instr_codes = pandas.read_csv(os.path.join(code_file_dir, 
                                                           WODInstrumentCodes.WOD_INSTRUMENT_CODES_FILENAME), 
                                              skiprows=[1])
        self.xbt_codes = self.df_instr_codes[self.df_instr_codes.Instrument.apply(lambda x: x.split(':')[0] == WODInstrumentCodes.XBT_PREFIX)]
        self.code_to_name_dict = {r1[1]: r1[2] for r1 in self.xbt_codes.to_records()}
        self.name_to_code_dict = {r1[2]: r1[1] for r1 in self.xbt_codes.to_records()}
    
    def code_to_name(self, wod_code):
        try:
            wod_name = self.code_to_name_dict[wod_code]
        except KeyError:
            wod_name = self.code_to_name_dict[0]
        return 
    
    def name_to_code(self, wod_name):
        try:
            wod_code = self.name_to_code_dict[wod_name]
        except KeyError:
            wod_code = 0
        return wod_code
    
def get_wod_encoders():
    return {
        'instrument': WODInstrumentCodes
    }
    
    
    