"""
Performs different exploratory analysis on the original data source, to help scientists in understanding the data 
"""

import argparse
from dataexploration import plot_unknown_brand_probes
def main():

    parser = argparse.ArgumentParser(description='Performs different exploratory analysis on the original data source, to help scientists in understanding the data')
    parser.add_argument('--path',default='./',help='input train and test files location')
    parser.add_argument('--outpath',default='./',help='output train and test files location')
    parser.add_argument('--plot_unknown_brand',default='yes', help='Represent XBTs data instances as latitude-vs-longitude plots, labeling data points with respect cruis ID and platform')
    args = parser.parse_args()

    if args.plot_unknown_brand == 'yes':
        message = 'Reading data from {}, storing plotted  results in {}'.format(args.path, args.outpath)
        print(message)
        plot_unknown_brand_probes(args.path, args.outpath)

if __name__ == "__main__":
    # execute only if run as a script
    main()
