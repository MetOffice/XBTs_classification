import datetime

DATESTAMP_TEMPLATE = '{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}'

def generate_datestamp():
    return DATESTAMP_TEMPLATE.format(dt=datetime.datetime.now())
