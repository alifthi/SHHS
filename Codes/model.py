import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers as ksl
import numpy as np

class model:
    def __init__(self):
        self.inputNames = ['abdo', 'air', 'chin', 'chst', 'eeg1', 'eeg2', 'eogl', 'eogr', 'hr']
        self.freq = [10,2,3,4,5,6,7,8,9]
        self.len = 1
        self.net = self.buildModel()
    def buildModel(self,transfer = False):
        # define inputs of model
        inputs = {}
        for f,name in enumerate(self.inputNames):
            inputs[name+'net'] = ksl.Input(shape = [1,self.len*self.freq[f]])
            outputs={}
        for name in inputs.keys():
            x = ksl.Conv1D(64,kernel_size = 5,strides = 1,padding = 'same')(inputs[name])
            x = ksl.MaxPooling1D(2,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(32,kernel_size = 5,strides = 1,padding = 'same')(x)
            x = ksl.MaxPooling1D(2,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(16,kernel_size = 5,strides = 1,padding = 'same')(x)
            x = ksl.MaxPooling1D(2,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(16,kernel_size = 5,strides = 1,padding = 'same')(x)
            x = ksl.MaxPooling1D(2,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            outputs[name] = ksl.Resizing(height = 16,width = 1024)(x) 

    def transformer(self):
        pass
    def trainModel(self):
        pass
    def plotHist(self):
        pass
    def callBacks(self):
        pass
    def test(self):
        pass
    def train(self):
        pass
net = model()
print(net.net)