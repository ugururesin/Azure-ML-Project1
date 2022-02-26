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
Scikit-learn pipeline is pretty straightforward and self-explanatory. For this we wrote a script called `train.py` where we defined a datastore with `TabularDatasetFactory` to read delimitted file. Then there was a `clean_data` function to

  - Removing NA's from the data
  - Doing encode for categorical values
  - Transforming values to desired form and others.
  
After doing the cleaning part we splitted the data to training and test. We used `Logistic Regression` as a classification algorithm and defined arguments `c` and `max_iter` so that we can experiment hyper parameters with hyperdrive. 

As a parameter sampler I used `RandomParameterSampling` with `uniform` and `quniform` where I had the opportunity to uniformly sampling from min `0` to max `20` for `c` hyperparameter and for `max_iter` I could set min `10` to max `250` with increment value `1`. 

The reason behind choosing `RandomParameterSampling` is that it is way faster than other samplers as it doesn't require pre-specified values for its search space and can traverse randomly to find the optimal value which often finds the ideal one which leads to early termination.

I defined the `BanditPolicy` with `evaluation_interval=1`, `delay_evaluation=5`, `slack_factor = 0.2` because it terminates all the models worse than the current best model based on the prmiary metrics. This is slightly better with flexibility than other policies.

With this pipeline I got accuracy `0.9153262518968134` for this dataset.

## AutoML
Using `Automl` I found most intuitive, I just defined the data store with `TabularDatasetFactory` and passed the data to `clean_data` function and concated the data to `pandas` dataframe. After that defined the `automl_config` for `tasks`, `primary_metric` and others and then experiment the run. With logs I could see all the pipeline steps, models and their metrics. Then I could also get the best child to save the best model.

## Pipeline comparison
| Model | Accuracy |
|-|-|
| Hyperdrive | 0.9153262518968134 |
| Automl | 0.9161 |

I believe difference is trivial in accuracy and if we talk about architecture `automl` is intuitive and it uses several preprocessing steps with model variations where in hyperdrive we only tune logistic regression with predefined processing. For more imbalanced and unseen data automl would do well then hyperdrive.

## Future work
Future improvements need borader exposer to hyperdrie and automl. We could only use one algorithm for `hyperdrive` exploration where we had time constraint for `automl`. Besides automated feature engineering can be done to see if it does better than the `clean_data` function. May be another imbalanced data set can be tested with above setup as well. 
