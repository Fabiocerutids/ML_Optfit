import optuna 
from optuna.samplers import TPESampler
import random 
import numpy as np
from ml_optfit.nn_model_creation import Build_NN 
import tensorflow as tf
from collections import defaultdict
import warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperOptim():
    def __init__(self, 
                 direction, 
                 train, 
                 valid, 
                 features, 
                 target, 
                 evaluation_func, 
                 prediction_type='classification',
                 seed=42):
        self.direction=direction
        self.x_train = train[features]
        self.x_valid = valid[features]
        self.y_train = train[target]
        self.y_valid = valid[target]
        self.direction=direction
        self.evaluation_func=evaluation_func
        self.SEED=seed
        self.prediction_type=prediction_type
        self.best_metric=0
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
        model.fit(self.x_train, self.y_train)
        if self.prediction_type=='classification':
            predict = model.predict_proba(self.x_valid)
            score, best_threshold = self._optimize_thresholds(predict)
            optuna_dict['best_threshold']=best_threshold
        else:
            predict = model.predict(self.x_valid)
            score = self.evaluation_func(self.y_valid, predict) 
        if score > self.best_metric:
            self.best_metric=score
            self.best_hyperparams = optuna_dict
        return score
    
    def _optimize_thresholds(self, pred):
        max_score = 0
        best_thresh = 0
        for i in np.linspace(0,1,num=100):
            new_pred=pred.copy()
            new_pred = new_pred[:,1]
            new_pred[new_pred>i]=1
            new_pred[new_pred<=i]=0
            if self.evaluation_func(self.y_valid, new_pred) > max_score:
                max_score = self.evaluation_func(self.y_valid, new_pred)
                best_thresh = i
        return max_score, best_thresh
    
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
        return self.study, self.best_hyperparams
    
    
class HyperOptimNN():
    def __init__(self, 
                 direction, 
                 train, 
                 valid, 
                 y_valid,
                 evaluation_func,
                 loss_func,  
                 prediction_type='classification',
                 seed=42,
                 epochs=200,
                 patience=10): 
        self.direction=direction
        self.loss_func = loss_func
        self.direction=direction
        self.evaluation_func=evaluation_func
        self.train = train 
        self.valid = valid
        self.y_valid = y_valid
        self.epochs=epochs
        self.prediction_type=prediction_type
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, restore_best_weights=True)
        self.SEED=seed
        self.best_metric = 0
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        
    def _objective_func_nn(self, trial):
        optuna_dict = self._get_optuna_dict_nn(trial)
        model = Build_NN(param_dict=optuna_dict).build()
        model.compile(optimizer='adam', loss=self.loss_func)
        model.fit(self.train, validation_data = self.valid, epochs=self.epochs, callbacks=[self.callback], verbose=0)
        predict = model.predict(self.valid, verbose=0)
        if self.prediction_type=='classification':
            score, best_threshold = self._optimize_thresholds(predict)
            optuna_dict['best_threshold']=best_threshold
        else:
            score = self.evaluation_func(self.y_valid, predict) 
        if score > self.best_metric:
            self.best_metric=score
            self.best_hyperparams = optuna_dict
        return score
    
    def _apply_optuna_int(self, trial, target_dic,k, subk, i):
        if 'step' in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], step=target_dic[k][subk]['step'], log=target_dic[k][subk]['log'])
        elif 'step' in target_dic[k][subk].keys() and 'log' not in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], step=target_dic[k][subk]['step'])
        elif 'step' not in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], log=target_dic[k][subk]['log'])
        else:
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'])
        
    def _apply_optuna_int_units(self, trial, target_dic,k, subk, current_max, i):
        if 'step' in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=current_max, step=target_dic[k][subk]['step'], log=target_dic[k][subk]['log'])
        elif 'step' in target_dic[k][subk].keys() and 'log' not in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=current_max, step=target_dic[k][subk]['step'])
        elif 'step' not in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=current_max, log=target_dic[k][subk]['log'])
        else:
            return trial.suggest_int(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=current_max)
    
    def _apply_optuna_float(self, trial, target_dic,k, subk, i):
        if 'step' in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_float(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], step=target_dic[k][subk]['step'], log=target_dic[k][subk]['log'])
        elif 'step' in target_dic[k][subk].keys() and 'log' not in target_dic[k][subk].keys():
            return trial.suggest_float(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], step=target_dic[k][subk]['step'])
        elif 'step' not in target_dic[k][subk].keys() and 'log' in target_dic[k][subk].keys():
            return trial.suggest_float(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'], log=target_dic[k][subk]['log'])
        else:
            return trial.suggest_float(k+'-'+subk+str(i), low=target_dic[k][subk]['low'], high=target_dic[k][subk]['high'])
    
    def _apply_optuna_str(self, trial, target_dic, k, subk, i):
        return trial.suggest_categorical(k+'-'+subk+str(i), target_dic[k][subk]['vals'])
    
    def _get_optuna_dict_nn(self, trial):
        out_dict = defaultdict(defaultdict)
        for target_dic in [self.input_hyper, self.common_hyper, self.output_hyper]:
            for k in target_dic.keys():
                units_l = []
                units_val_sel = []
                activation_l = []
                dropouts_l = [] 
                for subk in target_dic[k].keys():
                    if subk == 'n_hidden_layers':
                        out_dict[k][subk] = self._apply_optuna_int(trial, target_dic,k, subk, 0)
                    elif type(target_dic[k][subk]) == dict:   
                        for i in range(out_dict[k]['n_hidden_layers']):
                            if subk=='units':
                                if len(units_val_sel)>0:
                                    optuna_val =self._apply_optuna_int_units(trial, target_dic,k, subk, units_val_sel[-1], i)
                                    units_l.append(2**optuna_val)
                                    units_val_sel.append(optuna_val)
                                else:
                                    optuna_val = self._apply_optuna_int(trial, target_dic,k, subk, i)
                                    units_l.append(2**optuna_val)
                                    units_val_sel.append(optuna_val)
                            elif subk=='activation': 
                                activation_l.append(self._apply_optuna_str(trial, target_dic,k, subk, i))   
                            elif subk=='dropouts':
                                dropouts_l.append(self._apply_optuna_float(trial, target_dic,k, subk, i))
                    else:
                        out_dict[k][subk]=target_dic[k][subk]
                out_dict[k]['units']=units_l
                out_dict[k]['activation']=activation_l
                out_dict[k]['dropouts']=dropouts_l
        return out_dict 
    
    def _optimize_thresholds(self, pred):
        max_score = 0
        best_thresh = 0
        for i in np.linspace(0,1,num=100):
            new_pred=pred.copy()
            new_pred = new_pred
            new_pred[new_pred>i]=1
            new_pred[new_pred<=i]=0
            if self.evaluation_func(self.y_valid, new_pred) > max_score:
                max_score = self.evaluation_func(self.y_valid, new_pred)
                best_thresh = i
        return max_score, best_thresh
    
    def optimize_nn(self, 
                    input_hyper, 
                    common_hyper, 
                    output_hyper, 
                    study_name,  
                    multivariate=True, 
                    n_trials=50, 
                    timeout=None, 
                    load_if_exists=True, 
                    n_jobs=-1, 
                    show_progress_bar=True):
        self.sampler = TPESampler(seed=self.SEED, multivariate=multivariate)
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler, study_name=study_name, load_if_exists=load_if_exists)
        self.input_hyper = input_hyper
        self.common_hyper = common_hyper
        self.output_hyper = output_hyper
        self.study.optimize(self._objective_func_nn, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar, timeout=timeout)
        return self.study, self.best_hyperparams
    
"""
{'input_1':{
            'input_shape':(10,),
            'n_hidden_layers':{'type':'int', 'low':2, 'high':4},
            'units':[64,32],
            'activation':{'type':'class', 'vals':['relu', 'relu']},
            'dropouts':{'type':'float', 'low':0.1, 'high':0.2},
            },
'input_2':{},
'common':{'n_hidden_layers':{'type':'int', 'low':2, 'high':4},
            'units':{'type':'int', 'low':2, 'high':4},
            'activation':{'type':'class', 'vals':['relu', 'relu']},
            'dropouts':{'type':'float', 'low':0.1, 'high':0.2},
            },
'output_1':{
            'n_outputs':1,
            'n_hidden_layers':{'type':'int', 'low':2, 'high':4},
            'units':{'type':'int', 'low':2, 'high':4},
            'activation':{'type':'class', 'vals':['relu', 'relu']},
            'dropouts':{'type':'float', 'low':0.1, 'high':0.2}
            },
'output_2':{}
}
"""