.. _preprocessing_API:

Preprocessing API
=================

.. automodule:: preprocessing.preprocessor
  :members:

Basic usage is

.. code-block:: console

  $ python2.7 -m preprocessing.preprocessor TRAIN_DATASET_NAME TEST_DATASET_NAME

further options are provided by typing :code:`python2.7 -m preprocessing.preprocessor -h`

.. code-block:: console

  $ python2.7 -m preprocessing.preprocessor -h
  usage: preprocessor.py [-h] [--path PATH] [--outpath OUTPATH]
                         [-startyear STARTYEAR] [-endyear ENDYEAR]
                         [-useless_features USELESS_FEATURES [USELESS_FEATURES ...]]
                         [-features_to_get_labeled FEATURES_TO_GET_LABELED [FEATURES_TO_GET_LABELED ...]]
                         [-features_to_get_dummy FEATURES_TO_GET_DUMMY [FEATURES_TO_GET_DUMMY ...]]
                         [-reshuffle RESHUFFLE] [-train_split TRAIN_SPLIT]
                         [-test_split TEST_SPLIT]
                         train test

  Redefine train-test split. Apply features engineering. Produce a train/set
  dataset for each year

  positional arguments:
    train                 training set name
    test                  test set name

  optional arguments:
    -h, --help            show this help message and exit
    --path PATH           input train and test files location
    --outpath OUTPATH     output train and test files location
    -startyear STARTYEAR  start year
    -endyear ENDYEAR      end year
    -useless_features USELESS_FEATURES [USELESS_FEATURES ...]
                          features to be removed
    -features_to_get_labeled FEATURES_TO_GET_LABELED [FEATURES_TO_GET_LABELED ...]
                          categorical features to be converted into ordinal
                          features
    -features_to_get_dummy FEATURES_TO_GET_DUMMY [FEATURES_TO_GET_DUMMY ...]
                          categorical features to be converted into dummy
                          variables
    -reshuffle RESHUFFLE  boolean flag to activate reshuffling
    -train_split TRAIN_SPLIT
                          proportion of original data to be retained for
                          training
    -test_split TEST_SPLIT
                          proportion of original data to be retained for testing
