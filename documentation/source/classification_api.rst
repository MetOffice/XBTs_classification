.. _classification_API:

Classification API
==================

It provides modules that perform single and multiple machine learning experiments

Single classification experiment
--------------------------------

.. automodule:: classification.classification
  :members: run_experiment

It can be called by a :code:`main()` function using the following command

.. code-block:: console

  $ python2.7 -m   classification.classification TRAIN_SET_NAME TEST_SET_NAME JSON_DESCRIPTOR

further details can be obtaine using the :code:`-h` flag

.. code-block:: console

  usage: classification.py [-h] [--path PATH] [--outpath OUTPATH] [--year YEAR]
                          train test json_descriptor

  Run a specific classification experiment

  positional arguments:
    train              training set name
    test               test set name
    json_descriptor    json descriptor file name

  optional arguments:
    -h, --help         show this help message and exit
    --path PATH        input train and test files location
    --outpath OUTPATH  outputpath for results
    --year YEAR        year

Multiple classification experiments
-----------------------------------

.. automodule:: classification.classification_pipeline
  :members:

Basi usage is:

.. code-block:: console

  $ python2.7 -m   classification.classification_pipeline TRAIN_SET_NAME TEST_SET_NAME JSON_DESCRIPTORS_FOLDER

further details can be obtaine using the :code:`-h` flag

.. code-block:: console

  usage: classification_pipeline.py [-h] [--path PATH] [--outpath OUTPATH]
                                  [--year YEAR]
                                  train test json_descriptors_folder

  Runs several machine learning experiments for a given year of analysis

  positional arguments:
    train                 training set name
    test                  test set name
    json_descriptors_folder path to the folder containing json descriptors

    optional arguments:
    -h, --help            show this help message and exit
    --path PATH           input train and test files location
    --outpath OUTPATH     outputpath for results
    --year YEAR           year
