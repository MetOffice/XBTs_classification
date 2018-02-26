User Guide
==========

Welcome the the **XBTs_classifixation** system user guide. In the following, the basic functionalities of the system will be explained,
along with instructions for running single (or multiple) classification experiments.

The system allows users to

* Define customized :ref:`train-test splits <data_splitting>` from original data source 
* Design machine learning :ref:`experiments <experiment_design>` in a simple and practical way
* :ref:`Run <run_experiments>` the experiments 
* :ref:`Collecting and plotting <post_processing>` the numerical outcome of the performed experiments

.. _data_splitting:

Data Splitting
--------------

Defining a proper train-test split is crucial for properly training a learning algorithm and testing its generalization properties.
The :ref:`preprocessing_API` contained in the system allows to easily perform such procedure.
Here is an example of how to use it: suppose your original train and test set data are contained in a folder called :code:`source_folder`

.. code-block:: console
   
   $ls ${HOME}/source_folder
   train.csv test.csv

and you want to generate a new set of train a test data, where:

* the feature :code:`cruise_ID` is discarded
* you want to collapse train and test set together, reshuffle the data and split with a 60-40 ratio
* you want to create a collection of train and test set for all the years running from 1991 up to 1993, into a new folder called :code:`out_folder`

 This can be easily achieved by typing

.. code-block:: console
   
   $python 2.7 -m preprocessing.preprocessor train.csv test.csv --path ${HOME}/source_folder --outpath  ${HOME}/out_folder -startyear 1991 -endyear 1993 -useless_features cruise_ID -reshuffle True -train_split .6 -test_split .4
   $ls ${HOME}/out_folder
   test_1991.csv test_1992.csv test_1993.csv train_1991.csv train_1992.csv train_1993.csv

for more detailed information, look at the  :ref:`preprocessing_API` guide.

.. _experiment_design:

Designing an experiment
-----------------------

Machine learning experiments can be designed by filling simple json descriptors with the following structure

.. code-block:: json

   {"input_features":{"useless_features":["temperature_profile", "date", "depth_profile", "institute", "country"],
                      "features_to_get_labeled":[],
                     "features_to_get_dummy":["platform"],
                     "features_to_rescale":["max_depth","lat","lon", "platform"]},
    "output_features":["instrument_type", "instrument_type_and_manifacturer"],
    "operations":{"generate_target_outputs":{"module":"preprocessing.features_engineering",
                                          "function":"probe_type_output",
                                          "inputs":{"column":"instrument"}},
                  "map_zero_platform_to NaN":{"module":"preprocessing.features_engineering",
                                              "function":"map_attribute_value_to_NaN",
                                              "inputs":{"column":"platform",
                                                        "operator":"==",
                                                        "target_value":"\"O\""}},
                  "negative_max_depth_to_NaN":{"module":"preprocessing.features_engineering",
                                               "function":"map_attribute_value_to_NaN",
                                               "inputs":{"column":"max_depth",
                                                         "operator":"<",
                                                         "target_value":"0"}}},
    "rescale_all":1,
    "learner":{"module_name":"sklearn.neighbors",
               "python_class":"KNeighborsClassifier"},
    "tuning":{"param_grid":{"n_neighbors":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 20],
                            "weights":["distance"],
                            "n_jobs":[-1]},
              "scoring":"accuracy",
              "n_jobs":1,
              "cv":3,
              "return_train_score":false}

let us analyze more in detail what the different keys of this json file mean.

1. :code:`"input_features"`: it defines the collection of features representing the input for a chosen machine learning algorithm. Its subkeys define:

 * :code:`"input_features"`: features that have not been removed during the preprocessing, but will removed during the data manipulation phase.
 * :code:`"features_get_labeled"`: categorical features to be transformed into ordinal integers.
 * :code:`"features_to_get_dummy"`: categorical features to be transformed into dummy variables.
 * :code:`"features_to_rescale"`: numerical features you want to get rescaled before starting the learning.

2. :code:`"output_features"`: it defines the collection of output targets. Some of them could be already existing features of the dataset, others can get created by defining proper features engineering operations.
3. :code:`"operations"`: this is a quite important section of the descriptor. It defines the features engineering operations to be applied on the data. Each operation is labeled by a corresponding subkey: in the example above they are
 
 * :code:`"generate_target_outputs"`: it takes information from the original dataset for defining a new output target.
 * :code:`"map_zero_platform_to NaN"`: it sets to NaN every instance of the :code:`platform` label that is equal to :code:`"O"`
 * :code:`"negative_max_depth_to_NaN"`: it sets to negative values of :code:`max_depth` input feature to NaN.

 Each operation is generated by a method defined in the :ref:`features engineering  <features_engineering>` module, by defining proper input parameters.
 Users can introduce new features engineering operations by adding properly tested methods into such module.
 If the methods have been robustly tested and are bug-free, then the system will take care of executing them before starting the learning procedure.

4. :code:`"rescal_all"`: to allow newly created dummy variables to be rescaled.
5. :code:`"learner"`: the learning algorithm you want to adopt. You will just need to define the :code:`"module_name"` containing the :code:`"python_class"` that implement the given algorithm. Information about the paramaters needed for instantiating the class have to be provided in the dictionary defined by the :code:`"tuning"` key.
6. :code:`"tuning"` : defines the grid of values to be used for tuning the algorithm hyperparameters, it contains the following subkeys:
 
 * :code:`"param_grid"`: dictionary used to instantiate the learning class, where hyperparameters are defined through list of values.
 * :code:`"scoring"`: the metric used for optimizing the hyperparameters values.
 * :code:`"cv"`: number of folds applied during K-fold cross validation

 The tuning is performed by using the  :code:`GridSearchCV` class of the `Scikit-learn <http://scikit-learn.org/stable/>`_ library. More information about how to choose the paramaters for instantiating this class can be found `here <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_ . It is recommended not to set to :code:`true` the value of :code:`"return_train_score"`, since this will considerably slow down the tuning procedure.

Many different machine learning experiments can be easily designed using json descriptors as the one shown above. 
Once you have finished to design one, or more experiments, you are ready to run the classification procedure.

.. _run_experiments:

Running classification experiments
----------------------------------

Classification experiments can be performed by using the :ref:`classification_API`. Let's say you want to run a single specific experiment for year :code:`YYYY`, described by a :code:`${HOME}/json_descriptors/experiment_X.json`, your data are contained in :code:`${HOME}/source_folder`, and you want to create results in :code:`${HOME}/out_folder` then just type

.. code-block:: console

  $ python2.7 -m   classification.classification train_YYYY.csv test_YYYY.csv experiment_X.json --path ${HOME}/source_folder --outpath ${HOME}/out_folder
 
results will be stored in a new folder called :code:`${HOME}/out_folder/LEARNER_CLASS_NAME`. For example, if you have used the k-nearest-neighbors algorithm, this is what you should get

.. code-block:: console

  $ ls ${HOME}/out_folder/
  KNeighborsClassifier
  $ ls ${HOME}/out_folder/KNeighborsClassifier
  experiment_X
  $ ls ${HOME}/out_folder/KNeighborsClassifier/experiment_X
  YYYY_instrument_type_and_manifacturer_prediction.json  YYYY_instrument_type_tuning.json 
  YYYY_instrument_type_and_manifacturer_tuning.json      
  YYYY_instrument_type_prediction.json                   

for each year and output target, the system creates json descriptors containing:

* information about predicted probabilities, accuracy, recall scores and other useful metrics
* information about hyperparameters tuning

If you want to run several different experiments for a given year then you can use the following command

.. code-block:: console

  $ python2.7 -m   classification.classification_pipeline train test ${HOME}/json_descriptors --year YYYY --path ${HOME}/source_folder --outpath ${HOME}/out_folder

which will call the single experiment classification script several times (depending on how many json descriptors have been created)
Depending on the learning algorithm adopted and the data set size, it could become computationally expensive to run many experiments on your desktop computer. That is why the system includes also scripts for running the analysis on cluster machines.

.. _post_processing:

Post Processing
---------------

Once you all the experiments have been performed, you are ready to collect results and plot them.
