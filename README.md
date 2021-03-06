# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, an Azure ML pipeline is built and optimized using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.  
(Reference: https://archive.ics.uci.edu/ml/datasets/bank+marketing)

To get the best classification model, two approaches were conducted: `hyperdrive` and `automl`.  
Details are given below.

## Pipeline
![Pipeline Diagram](pipeline.jpg?raw=true)

## Scikit-learn Pipeline
Scikit-learn pipeline is pretty straightforward and self-explanatory. For this we wrote a script called `train.py` where we defined a datastore with `TabularDatasetFactory` to read delimitted file. Then there was a `clean_data` function that conduct the following data wrangling operations:

1. Removing NA's from the data  
2. Doing encode for categorical values  
3. Transforming values to desired form and others
  
After completing the cleaning part, the data is splitted into training and test. As a classification algorithm `Logistic Regression` is used, then `c` and `max_iter` parameters were defined in order to experiment hyper parameters with hyperdrive. 

As a parameter sampler, the `RandomParameterSampling` is used with the following parameters:  
```
RandomParameterSampling(
    {
        '--C' : choice(0.01,0.1,1,5,20,100),
        '--max_iter': choice(10,50,100)
    }
)
```

`RandomParameterSampling` provides faster sampling due to the fact that it does not require pre-specified values for its search space and can traverse randomly to find the optimal value which often finds the ideal one which leads to early termination.

The `BanditPolicy` is defined with evaluation_interval = 1, slack_factor= 0.1 in order to terminates all the models worse than the current best model based on the prmiary metrics.

![hyperdrive1](results_hyperdrive.JPG?raw=true)

This pipeline yielded the following accuracy score: 0.91760243

## AutoML
In this approach, the models (there are several ML algorithms trained) and hyperparameters are generated automatically based on the following automl configuration:

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=15,
    task='classification',
    enable_early_stopping = True,
    primary_metric='AUC_weighted',
    training_data=ds,
    label_column_name='y',
    enable_onnx_compatible_models=True,
    n_cross_validations=3,
    compute_target = compute_target)
```

**experiment_timeout_minutes**: Defines how long an experiment will take, in minutes. 15 used here.

**task**: Specifies whether the model will 'classification' or 'regression'.

**primary_metric**: Defines the metric for the model performance. Here AUC is used.

**label_column_name"**: Defines the dependent variable (label).

**enable_onnx_compatible_models**: ONNX stands for 'Open Neural Network Exchange' that allows to deploy models in different frameworks and platforms.

**n_cross_validation**: The number of cross validations to conduct. Here 3 is used. Therefore, metrics are calculated based on the average of 3 validation scores.

Using `Automl` several algorithms are fitted and the **METRIC**s (the result of computing score on the fitted pipeline) are compared as shown in the table below.

(Run details are available in the notebook)

![automl_table](results_automl.JPG?raw=true)

## AutoML Model Parameters
The AutoML yielded the best model as a **VotingEnsemble** model with the **following parameters**:

  * 'run_algorithm': 'VotingEnsemble'  
  * 'ensembled_iterations': '[23, 0, 25, 1, 15, 21, 20]'  
  * 'ensembled_algorithms': "['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'LogisticRegression']"  
  * 'ensemble_weights': '[0.26666666666666666, 0.26666666666666666, 0.06666666666666667, 0.06666666666666667, 0.2, 0.06666666666666667, 0.06666666666666667]'  
  * 'best_individual_pipeline_score': '0.9471770965872123'  
  * 'best_individual_iteration': '23'  
  * 'model_name': 'AutoML5540d998b26'  
  * 'score': '0.948525333973904'

As shown above, the model is an ensemble ML model of the following models 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'LogisticRegression' with the following weightings 0.26666666666666666, 0.26666666666666666, 0.06666666666666667, 0.06666666666666667, 0.2, 0.06666666666666667, 0.06666666666666667 respectively.
 

## Pipeline comparison
| Model | Accuracy |
|-|-|
| Hyperdrive | 0.91760243 |
| Automl | 0.9485 |

First of all, it should be noted that the accuries always be different due to randomization in data sampling and fitting. Based on the results, the difference between Hyperdrive and Automl is worth to consider however it should be considered that the number of iterations was limited to due to time constraint. In general, 'Automl' is expected to yield better results especially in the case of the data is imbalanced.


## Future Improvements
In order to improve the performance metric (here AUC_weighted is used), further iterations would be conducted. To do achieve that, ```n_cross_validation```  and/or ```experiment_timeout_minutes``` might be increased. However, it should be noted that there is a trade off between training cost (time and computation cost) and the performance metric. Thus, experiementations need to be conducted carefully. 

In addition, the dataset is imbalanced as shown below.

![labal_dist](label_distribution.png?raw=true)

It's quite usual in real life to get an imbalanced data however the dataset should be as balanced as possible. To address this issue, some of techniques are given below.  
1.) Upsampling Minority Class  
2.) Downsampling Majority Class  
3.) Generate Synthetic Data  
4.) Combine Oversampling and Undersampling Techniques  
5.) Balanced Class Weight  

## Proof of Compute Deletion
You may also check the notebook!  
![delcompute](proof_of_compute_deletion.JPG?raw=true)
![delcompute2](proof_of_compute_deletion2.JPG?raw=true)

