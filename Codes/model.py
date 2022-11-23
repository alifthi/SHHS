'''
The MIT License (MIT)

Copyright (c) 2022 Ali Fathi Jahromi <alifathi8008@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
        kernelSize = 5
        poolingSize = 2
        strides2D = 1
        poolingSize2D = 2
        ReLURate = 0.01
        dropoutRate = 0.01
        for name in self.inputNames:
            # define first networks of every inputs of model
            print(name)
            r = self.freq[name] 

            inputs[name+'Net'] = ksl.Input(shape = [None,self.len*r])

            x = ksl.Conv1D(8,kernel_size = kernelSize,padding = 'same')(inputs[name+'Net'])
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(16,kernel_size = kernelSize,padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(32,kernel_size = kernelSize,padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)

            
            outputs[name] = x 
        # define shared model
        concatLayer = ksl.concatenate(list(outputs.values()),axis = -1)

        x = ksl.Conv1D(64,kernel_size = 3,strides = strides2D,padding = 'same')(concatLayer)
        x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        # x = ksl.Conv1D(64,kernel_size = 3,strides = strides2D,padding = 'same')(concatLayer)
        # x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        # x = ksl.BatchNormalization()(x)

        x = ksl.Conv1D(32,kernel_size = 3,strides = strides2D,padding = 'same')(x)
        x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv1D(16,kernel_size = 3,strides = strides2D,padding = 'same')(x)
        x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)
        
        # x = ksl.Dense(512)(x)
        # x = ksl.LeakyReLU(ReLURate)(x)
        # x = ksl.Dropout(0.2)(x)

        # x = ksl.Dense(256)(x)
        # x = ksl.LeakyReLU(ReLURate)(x)
        # x = ksl.Dropout(0.1)(x)

        x = ksl.Dense(128)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(0.1)(x)

        x = ksl.Dense(64)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(0.1)(x)

        x = ksl.Dense(32)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        
        output = ksl.Dense(1,activation = 'sigmoid')(x)
        return tf.keras.Model(inputs = list(inputs.values()),outputs = output)
    def transformer(self):
        pass
    def compile(self):
        opt = optim.SGD(lr=0.01, decay=1e-6, momentum=0.9)  # we can use lr schedual
        Loss = loss.BinaryCrossentropy()
        self.net.compile(optimizer = opt,loss = 'binary_crossentropy',metrics = ['accuracy'])
        self.net.summary()
        
    def trainModel(self,signal=None,targets=None,validationData=None,validationTargets=None,batchSize = 64,epochs = 1):   
        hist = self.net.fit(signal,targets,epochs = epochs,batch_size = batchSize,
                    validation_data=[validationData,validationTargets])
        return hist
    def trainGenerator(self,util,inputNames,saveAddr,valData):
        normalLen = 30
        patientLen = 30
        numFiles = 830
        epochs = 10
        steps = int(numFiles/(normalLen+patientLen))
        for i in range(epochs):
            print(f'epoch {i}:\n')
            util.readSignals = [200077,200116,200081,200082,200093,200114,200115,200117]
            for j in range(steps):
                #try:
                trainD,trainT = util.dataGenerator(inputNames,normalLen = normalLen,patientLen = patientLen)
                print(f'step: {j}')
                #except:
                 #   break
                hist = self.net.fit(trainD,trainT,epochs=5,batch_size=128,validation_data=valData)
                del(trainD,trainT)
                with open(f"C:\\Users\\u23\\Documents\\project\\History2\\historys{j}e{i+2}.txt", 'w') as f: 
                    for key, value in hist.history.items(): 
                        f.write(f'epochs {i} :\n Key:  {key}  Value:  {value}')
                    f.write('\n####################################################')
            self.net.save(saveAddr+'model_' + str(i) + '.h5')
    @staticmethod
    def plotHist(Hist):
        from matplotlib import pyplot as plt
        plt.plot(Hist.history['accuracy'])
        plt.plot(Hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.show()
        plt.plot(Hist.history['loss'])
        plt.plot(Hist.history['val_loss'])
        plt.title('model loss')
        plt.show()
    def callBacks(self):
        pass
    def test(self):
        pass
    def loadModel(self,addr):
        self.net = tf.keras.models.load_model(addr)
        
