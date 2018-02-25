User Guide
==========

Welcome the the **XBTs_classifixation** system user guide. In the following, the basic functionalities of the system will be explained,
along with instructions for running single (or multiple) classification experiments.

The system allows users to

* Define customized train-test splits from original data source
* Design machine learning experiments in a simple and practical way
* Collecting and plotting the numerical outcome of the performed experiments

Data Splitting
--------------

Defining a proper train-test split is crucial for properly training a learning algorithm and testing its generalization properties.
The :ref:`preprocessing_API` contained in the system allows to easily perform such procedure
(EXAMPLE)

Design an experiment
--------------------

Machine learning experiments can be designed by filling simplle json descriptors with the following structure
(JSON DESCRIPTOR)

The system will ensure that input descriptors are written in the correct format.
Once you have finished to design one, or more experiments, you are ready to run the classification procedure

Running classification experiments
----------------------------------

Classification experiments can be performed by using the :ref:`classification_API`.
(EXAMPLE)

Post Processing
---------------

Once you all the experiments have been performed, you are ready to collect results and plot them.
