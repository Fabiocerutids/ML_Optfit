## Repository Description and Overview
The repository contains the code I developed to ease: Hyperparameter Tuning for Traditional ML, Neural Network Building and its Hyperparameter Optimization.

To automate hyperparameter search and NN Building, make sure to pass dictionaries with the structure provided in the tutorial folder.

Note: The repository exploits Optuna as the library of choice to perform hyperparameter search. To familiarize with the library: https://optuna.org/

## Installation Guide
To install the package:

```bash
pip install ml-optfit==0.1.0
```

Alternatively:

1. Clone the repository:
```bash
git clone https://github.com/Fabiocerutids/ML_Optfit.git
```

2. Locate yourself in the ML_Optfit folder:
```bash
cd ML_Optfit/
```

3. Run the following command in terminal:
```bash
pip install .
```

4. ML_Optfit is now installed, verify by running the command below:
```python
from ml_optfit.ml_optfit import HyperOptimNN
```

## How to use the package

### Traditional ML
```python
hyperopt=HyperOptim(direction='maximize', 
                    train=train, 
                    valid=valid, 
                    features=features, 
                    target='diabetes', 
                    evaluation_func=f1_score)

forest_hyper_dict = {'class_weight':{
                                    'type': 'class',
                                    'values': ['balanced', 'balanced_subsample', None]},
                    'n_estimators':{
                                    'type': 'int',
                                    'low': 100,
                                    'high':600,
                                    'log':False,
                                    'step':100},
                    'min_impurity_decrease':{
                                    'type': 'float',
                                    'low': 0,
                                    'high':0.1,
                                    'log':False,
                                    'step':0.01}
                                    }

study, best_hyper=hyperopt.optimize_model(model_type=RandomForestClassifier, 
                                         study_name='randomforest', 
                                         hyperparam_dict=forest_hyper_dict, 
                                         multivariate=False, 
                                         n_trials=30)
```

### Neural Networks
```python
opt_nn = HyperOptimNN(direction='maximize',
                      train=train_df.shuffle(buffer_size=1000).batch(500),
                      valid=valid_df.shuffle(buffer_size=1000).batch(500),
                      y_valid=valid[target].to_numpy(dtype=np.float32),
                      unshuffled_valid=valid_df.batch(500),
                      evaluation_func=f1_score,
                      loss_func='binary_crossentropy',
                      epochs=100)

input_hyper = {'input_1':{
                        'input_shape':(8,),
                        'n_hidden_layers':{'type':'int', 'low':1, 'high':3},
                        'units':{'type':'int', 'low':2, 'high':5},
                        'activation':{'type':'class', 'vals':['relu', 'tanh', 'selu']},
                        'dropouts':{'type':'float', 'low':0.01, 'high':0.9},
                        }}
common_hyper = {'common':{
                        'n_hidden_layers':{'type':'int', 'low':1, 'high':3},
                        'units':{'type':'int', 'low':2, 'high':5},
                        'activation':{'type':'class', 'vals':['relu', 'tanh', 'selu']},
                        'dropouts':{'type':'float', 'low':0.01, 'high':0.9},
                        }}
output_hyper = {'output_1':{
                            'n_outputs':1,
                            'n_hidden_layers':{'type':'int', 'low':1, 'high':3},
                            'units':{'type':'int', 'low':2, 'high':5},
                            'activation':{'type':'class', 'vals':['relu', 'tanh', 'selu']},
                            'dropouts':{'type':'float', 'low':0.01, 'high':0.9}
                            }}

study, best_hyper = opt_nn.optimize_nn(input_hyper=input_hyper,
                           common_hyper=common_hyper,
                           output_hyper=output_hyper,
                           study_name='TF Test',
                           n_trials=30, 
                           multivariate=False)
```
