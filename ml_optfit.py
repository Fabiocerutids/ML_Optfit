import optuna 
from optuna.samplers import TPESampler
import random 
import numpy as np
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperOptim():
    def __init__(self, 
                 direction, 
                 train, 
                 valid, 
                 features, 
                 target, 
                 evaluation_func, 
                 seed=42):
        self.direction=direction
        self.x_train = train[features]
        self.x_valid = valid[features]
        self.y_train = train[target]
        self.y_valid = valid[target]
        self.direction=direction
        self.evaluation_func=evaluation_func
        self.SEED=seed
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        
    def _get_optuna_dict(self, trial):
        """
        Hyperparemeter dict must have the following structure
        {'hyper_param_name1':
                            {'type': 'class',
                            'values': [...]},
        'hyper_param_name2':
                            {'type': 'int',
                            'low': 10,
                            'high':100,
                            'log':False,
                            'step':1},
        'hyper_param_name3':
                            {'type': 'float',
                            'low': 0.01,
                            'high':0.1,
                            'log':False,
                            'step':0.01}
                            }
        """
        optuna_dict = {}
        for k,v in self.hyperparam_dict.items():
            if v['type']=='class':
                optuna_dict[k] = trial.suggest_categorical(k, v['values'])
            elif v['type']=='int':
                if 'step' in v.keys() and 'log' in v.keys():
                    optuna_dict[k] = trial.suggest_int(k, low=v['low'], high=v['high'], step=v['step'], log=v['log'])
                elif 'step' in v.keys() and 'log' not in v.keys():
                    optuna_dict[k] = trial.suggest_int(k, low=v['low'], high=v['high'], step=v['step'])
                elif 'step' not in v.keys() and 'log' in v.keys():
                    optuna_dict[k] = trial.suggest_int(k, low=v['low'], high=v['high'], log=v['log'])
                else:
                    optuna_dict[k] = trial.suggest_int(k, low=v['low'], high=v['high'])
            elif v['type']=='float':
                if 'step' in v.keys() and 'log' in v.keys():
                    optuna_dict[k] = trial.suggest_float(k, low=v['low'], high=v['high'], step=v['step'], log=v['log'])
                elif 'step' in v.keys() and 'log' not in v.keys():
                    optuna_dict[k] = trial.suggest_float(k, low=v['low'], high=v['high'], step=v['step'])
                elif 'step' not in v.keys() and 'log' in v.keys():
                    optuna_dict[k] = trial.suggest_float(k, low=v['low'], high=v['high'], log=v['log'])
                else:
                    optuna_dict[k] = trial.suggest_float(k, low=v['low'], high=v['high'])
            else:
                raise Exception (f'The possible hyperparameter types are: [class, int, float], you provided {v[0]}')
        return optuna_dict
        
    def _objective_func(self, trial, model_type):
        optuna_dict = self._get_optuna_dict(trial)
        model = model_type(**optuna_dict)
        predict = model.predict(self.x_valid)
        return self.evaluation_func(self.y_valid, predict) 
    
    def optimize_model(self, 
                       model_type, 
                       study_name, 
                       hyperparam_dict, 
                       multivariate=True, 
                       n_trials=50, 
                       timeout=None, 
                       load_if_exists=True, 
                       n_jobs=-1, 
                       show_progress_bar=True):
        #Add storage for study
        self.sampler = TPESampler(seed=self.SEED, multivariate=multivariate)
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler, study_name=study_name, load_if_exists=load_if_exists)
        self.hyperparam_dict=hyperparam_dict
        self.study.optimize(lambda trial: self._objective_func(trial, model_type), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar, timeout=timeout)
        return self.study