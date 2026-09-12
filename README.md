# CDKPredict

This repository contains the code used for "***Interpretable clinicogenomic model for risk-adapted therapeutic intervention in metastatic breast cancer***", currently under revision.

CDKPredict represents the implementation of ***ReCAST*** to predict clinical outcomes in patients receiving first-line CDK4/6 inhibitors plus endocrine therapy. 
Please refer to https://github.com/mskcc/ReCAST for instructions on how to install and use ReCAST. 

In addition, we have provided in this repository the Python class used for implementing the scalable model (**CDKPredict-S**).  
CDKPredict-S expands on the default ReCAST framework to implement the mean risk-distribution contribution from each feature to mitigate the risk of distributional risk shifts derived from missing variables. 
The requirements and dependencies for running CDKPredict-S are the same as for ReCAST. 


## Specific implementation of ReCAST for predicting CDK4/6 inhibitors plus endocrine therapy outcomes

The rationale for developing ReCAST and defining the specific parameters during model initialization have been reported in "***Interpretable clinicogenomic model for risk-adapted therapeutic intervention in metastatic breast cancer***", currently under revision.

For initializing ReCAST for the genomic-only, clinical-only, and clinicogenomic model, as well as for the benchmark, the following parameters were set (here showed for CDKPredict, leveraging merged clinicogenomic data for prediction)

```python
CDKPredict = ReCAST(
    n_models = 100, 
    l1 = 1,
    n_folds=3, 
    bootstrap=True,
    random_state=94, 
    adaptive_lasso=True,                
    normalization=(0.01, 0.99), 
    auc_time_range=(3, 36),
    subagging=1,
    metric = 'auc'
)
```

For the scalable model:

```python
CDKPredictS = CDKPredict_scalable(
    n_models = 100, 
    l1 = 1, 
    n_folds=3, 
    bootstrap=True,
    random_state=94, 
    adaptive_lasso=True, 
    normalization = (0.01, 0.99), 
    metric = 'auc', 
    auc_time_range =(3, 36), 
    subagging = 1
)
```