'''
author: alifthi

Email: alifathi8008@gmail.com

lastEdit: 2022/NOV/3
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers as ksl
from tensorflow.keras import optimizers as optim
from tensorflow.keras import losses as loss
import numpy as np

class model:
    def __init__(self,inputNames):
        self.inputNames = inputNames
        self.freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}
        self.len = 30 # 32010
        self.net = self.buildModel()
    def buildModel(self):
        # define inputs of model
        inputs = {}                 # a dictionar that contain all inputs
        outputs = {}                # A dictionary that contain all outputs
        # define some hyper parametere
        d = 3
        kernelSize = 3
        poolingSize = 2
        strides2D = 1
        poolingSize2D = 2
        ReLURate = 0.1
        dropoutRate = 0.1
        for name in self.inputNames:
            # define first networks of every inputs of model
            print(name)
            r = self.freq[name] 

            inputs[name+'Net'] = ksl.Input(shape = [None,self.len*r])

            x = ksl.Conv1D(64,kernel_size = kernelSize,strides = int(np.ceil(r/d)),padding = 'same')(inputs[name+'Net'])
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(32,kernel_size = kernelSize,strides = int(np.ceil(r/d)),padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(16,kernel_size = kernelSize,strides = int(np.ceil(r/d)),padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(16,kernel_size = kernelSize,strides = int(np.ceil(r/d)),padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)

            x = ksl.Resizing(height = 1,width = 1024)(x) 
            outputs[name] = x 
        # define shared model
        concatLayer = ksl.concatenate(list(outputs.values()),axis = -1)

        x = ksl.Reshape((1,) + concatLayer.shape[1:])(concatLayer)
        x = ksl.Conv2D(64,kernel_size = 3,strides = strides2D,padding = 'same')(x)
        x = ksl.MaxPooling2D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv2D(32,kernel_size = 3,strides = strides2D,padding = 'same')(x)
        x = ksl.MaxPooling2D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv2D(16,kernel_size = 3,strides = strides2D,padding = 'same')(x)
        x = ksl.MaxPooling2D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)
        
        x = ksl.Flatten()(x)
        x = ksl.Dense(256)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(dropoutRate)(x)

        x = ksl.Dense(128)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(dropoutRate)(x)

        output = ksl.Dense(1,activation = 'sigmoid')(x)
        return tf.keras.Model(inputs = list(inputs.values()),outputs = output)
    def transformer(self):
        pass
    def compile(self):
        opt = optim.Adam(learning_rate = 0.01)  # we can use lr schedual
        Loss = loss.BinaryCrossentropy()
        self.net.compile(optimizer = opt,loss = Loss,metrics = ['accuracy'])
        self.net.summary()
        
    def trainModel(self,signal=None,targets=None,epochs = 1,batchSize = 125):   
        self.net.fit(signal,targets,epochs = epochs,batch_size = batchSize)
    def plotHist(self):
        pass
    def callBacks(self):
        pass
    def test(self):
        pass