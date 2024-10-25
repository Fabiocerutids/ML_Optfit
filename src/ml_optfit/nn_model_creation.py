import tensorflow as tf 
from keras import Input 
from keras.layers import Dense, Dropout, BatchNormalization, Concatenate 
from tensorflow.keras.models import Model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Build_NN():
    def __init__(self, param_dict, SEED=42):
        """
        Neural Network dict must have the following structure
        {'input_1':{
            'input_shape':(10, ),
            'n_hidden_layers':2,
            'units':[],
            'activation':[],
            'dropouts':[],
            },
        'input_2':{},
        'common':{},
        'output_1':{
            'n_outputs':10,
            'n_hidden_layers':2,
            'units':[],
            'activation':[],
            'dropouts':[]}},
        'output2':{}    
        """
        self.param_dict = param_dict 
        self.inputs = [i for i in self.param_dict.keys() if i[:5]=='input']
        self.input_layers = {}
        self.processed_inputs = {}
        self.input_n = 1 
        self.outputs = [i for i in self.param_dict.keys() if i[:6]=='output']
        self.output_layers = {}
        self.output_n = 1
        tf.random.set_seed(SEED)
        
    def _build_input(self, input_dict):
        layer = Input(shape=input_dict['input_shape'])
        self.input_layers[f'input_{self.input_n}']=layer
        self.input_n+=1
        layer = BatchNormalization()(layer)
        for k in range(input_dict['n_hidden_layers']):
            layer = Dense(input_dict['units'][k],
                          activation = input_dict['activation'][k])(layer)
            layer = Dropout(rate=input_dict['dropouts'][k])(layer)
            layer = BatchNormalization()(layer)
        self.processed_inputs[f'input_{self.input_n}']=layer
        
    
    def _build_common(self, common_dict):
        common_part = Concatenate()([v for v in self.processed_inputs.values()])
        common_part = BatchNormalization()(common_part)
        for k in range(common_dict['n_hidden_layers']):
            common_part = Dense(common_dict['units'][k],
                          activation = common_dict['activation'][k])(common_part)
            common_part = Dropout(rate=common_dict['dropouts'][k])(common_part)
            common_part = BatchNormalization()(common_part)
        self.common_part = common_part
    
    def _build_output(self, output_dict):
        out_layer = Dense(output_dict['units'][0],
                      activation = output_dict['activation'][0])(self.common_part)
        out_layer =  Dropout(rate=output_dict['dropouts'][0])(out_layer)
        out_layer = BatchNormalization()(out_layer)
        for k in range(1, output_dict['n_hidden_layers']):
            out_layer = Dense(output_dict['units'][k],
                          activation = output_dict['activation'][k])(out_layer)
            out_layer = Dropout(rate=output_dict['dropouts'][k])(out_layer)
            out_layer = BatchNormalization()(out_layer)
            
        out_layer = Dense(output_dict['n_outputs'], activation = output_dict['activation'][-1])(out_layer) #Creating the prediction
        self.output_layers[f'input_{self.output_n}']=out_layer
        self.output_n+=1
        
    def build(self):
        #Create Input Sections
        for inp in self.inputs:
            self._build_input(self.param_dict[inp])
        #Common Processing 
        self._build_common(self.param_dict['common'])
        #Create Outputs 
        for out in self.outputs:
            self._build_output(self.param_dict[out])
        model = Model(inputs=list(self.input_layers.values()),
                      outputs=list(self.output_layers.values()))
        return model