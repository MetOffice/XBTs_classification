import datetime

# flags for specifying how many files to split output over
OUTPUT_SINGLE = 'single'  # all output in one file
OUTPUT_YEARLY = 'yearly'  # output divided into files for each year
OUTPUT_MONTHLY = 'monthly'  # output divided into files for each month
OUTPUT_FREQS = [OUTPUT_SINGLE, OUTPUT_YEARLY, OUTPUT_MONTHLY]

DATESTAMP_TEMPLATE = '{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}'


def generate_datestamp():
    return DATESTAMP_TEMPLATE.format(dt=datetime.datetime.now())
