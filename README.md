# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains campaign data about prospective individuals for banks. Here we have information about clients and we are doing classification whether a client will subscribe to the bank service or not.

We followed two different approaches. One through `hyperdrive` and another with `automl`. Through hyperdrive we tried to find best hyperparameters `c` and `max_iter` which we found `3.5430558109339505` and `169.0` respectively with `0.9153262518968134` accuracy from run id `HD_a323d13a-c6a7-4160-a5af-e8444f840933_23`. And with automl we found `VotingEnsemble` with `0.9161` best accuracy where run id is `AutoML_1c1181c8-53c7-4470-a7ca-6bed4153fdb2_35`.

## Pipeline
![Pipeline Diagram](pipeline.jpg?raw=true)

## Scikit-learn Pipeline
Scikit-learn pipeline is pretty straightforward and self-explanatory. For this we wrote a script called `train.py` where we defined a datastore with `TabularDatasetFactory` to read delimitted file. Then there was a `clean_data` function that conduct the following data wrangling operations:

1. Removing NA's from the data  
2. Doing encode for categorical values  
3. Transforming values to desired form and others
  
After completing the cleaning part, the data is splitted into training and test. As a classification algorithm `Logistic Regression` is used, then `c` and `max_iter` parameters were defined in order to experiment hyper parameters with hyperdrive. 

As a parameter sampler, the `RandomParameterSampling` is used with the following parameters:  
**C** :(0.01,0.1,1,5,20,100)  
**max_iter** : (10,50,100)  

`RandomParameterSampling` provides faster sampling due to the fact that it does not require pre-specified values for its search space and can traverse randomly to find the optimal value which often finds the ideal one which leads to early termination.

The `BanditPolicy` is defined with evaluation_interval = 1, slack_factor= 0.1 in order to terminates all the models worse than the current best model based on the prmiary metrics.

![hyperdrive1](HyperDrive_run1.JPG?raw=true)
![hyperdrive2](HyperDrive_run1.JPG?raw=true)

This pipeline yielded the following accuracy score: 0.9176024279210926

## AutoML
Using `Automl` several algorithms are fitted and the **METRIC**s (the result of computing score on the fitted pipeline) are compared as shown in the table below.

![automl_table](automl_results.JPG?raw=true)

## Pipeline comparison
| Model | Accuracy |
|-|-|
| Hyperdrive | 0.9176 |
| Automl | 0.9471 |

First of all, it should be noted that the accuries always be different due to randomization in data sampling and fitting. Based on the results, the difference between Hyperdrive and Automl is worth to consider however it should be considered that the number of iterations was limited to due to time constraint. In general, 'Automl' is expected to yield better results especially in the case of the data is imbalanced.

## Proof of cluster clean up
![Cleanup](cleanup.JPG?raw=true)
