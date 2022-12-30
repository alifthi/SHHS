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
        self.len = 120 # 32010
        self.net = None # self.buildModel()
        self.dataFlag = True
        self.clbk = None
        self.trainData = []
        self.trainTargets = []
        self.testData = []
        self.testTargets = []
    def buildModel(self):
        # define inputs of model
        inputs = {}                 # a dictionar that contain all inputs
        outputs = {}                # A dictionary that contain all outputs
        # define some hyper parametere
        d = 3
        featureReluRate = 0.1
        featureDropoutRate = 0.1
        kernelSize = 3
        poolingSize = 2
        strides2D = 2
        poolingSize2D = 2
        ReLURate = 0.01
        dropoutRate = 0.2
        for name in self.inputNames:
            # define first networks of every inputs of model
            print(name)
            r = self.freq[name] 

            inputs[name+'Net'] = ksl.Input(shape = [self.len*r,1])

            x = ksl.Conv1D(16,kernel_size = kernelSize,strides = 2,activation = 'relu',padding = 'same')(inputs[name+'Net'])
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)
            
            x = ksl.Conv1D(8,kernel_size = kernelSize,strides = 2,activation = 'relu',padding = 'same')(x)
            x = ksl.MaxPooling1D(poolingSize,padding = 'same')(x)
            x = ksl.BatchNormalization()(x)

            outputs[name] = x 
        # define shared model
        concatLayer = ksl.concatenate(list(outputs.values()),axis = 1)

        x = ksl.Conv1D(64,kernel_size = 3,strides = strides2D,activation = 'relu',padding = 'same')(concatLayer)
        x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv1D(32,kernel_size = 3,strides = strides2D,activation = 'relu',padding = 'same')(x)
        x = ksl.MaxPooling1D(poolingSize2D,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)
        

        x = ksl.Flatten()(x)
        
        featureInput = ksl.Input(shape = [self.numFeatures])
        y = ksl.Dense(128,ativation = 'relu')(featureInput)
        y = ksl.LeakyRelu(featureReluRate)(y)
        y = ksl.Dropout(featureDropoutRate)(y)

        y = ksl.Dense(256,ativation = 'relu')(featureInput)
        y = ksl.LeakyRelu(featureReluRate)(y)
        y = ksl.Dropout(featureDropoutRate)(y)

        x = ksl.concatenate([x,y],axis = 0)
        x = ksl.Dense(32)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(dropoutRate)(x)

        x = ksl.Dense(16)(x)
        x = ksl.LeakyReLU(ReLURate)(x)
        x = ksl.Dropout(dropoutRate)(x)
        
        output = ksl.Dense(1,activation = 'sigmoid')(x)
        return tf.keras.Model(inputs = list(inputs.values()),outputs = output)
    def transformer(self):
        pass
    def compile(self,saveAddr):
        opt = optim.SGD(lr=0.1)  # we can use lr schedual
        Loss = loss.BinaryCrossentropy()
        self.net.compile(optimizer = opt,loss = 'binary_crossentropy',metrics = ['accuracy'])
        self.net.summary()
        
    def trainModel(self,signal=None,targets=None,validationData=None,validationTargets=None,batchSize = 64,epochs = 1):   
        hist = self.net.fit(signal,targets,epochs = epochs,batch_size = batchSize,
                    validation_data=[validationData,validationTargets],callbacks = self.clbk)
        return hist
    def trainGenerator(self,util,saveAddr,split = False):
        normalLen = 100
        patientLen = 100
        numFiles = 830
        epochs = 10
        steps = int(numFiles/patientLen)
        for i in range(epochs):
            print(f'epoch {i}:\n')

            for j in range(steps):
                if not self.dataFlag and j == len(self.trainData):
                    break
                if self.dataFlag:
                    try:
                        if split:
                            trainD,trainT,testD,testT = util.dataGenerator(normalLen = normalLen,
                                                                       patientLen = patientLen,split = True)
                            self.testData.append(testD)
                            self.testTargets.append(testT)
                        else:
                            trainD,trainT,testD,testT = util.dataGenerator(normalLen = normalLen,
                                                                         patientLen = patientLen)
                        self.trainData.append(trainD)
                        self.trainTargets.append(trainT)
                    except:
                        self.dataFlag = False
                        break

                     
                print(f'step: {j}')
                self.callBacks(epoch = 'epoch_' + str(i) + '_step_' +str(j))
                hist = self.net.fit(self.trainData[j],self.trainTargets[j],epochs=15,batch_size=128,
                                    validation_data=[self.testData,self.testTargets],callbacks = self.clbk)
                self.callBacks(epoch = 'epoch_' + str(i) + '_step_' +str(j))
                self.plotHist(hist,saveAddr = r'~/Documents/projects/SHHS/Plots',
                              i ='_epoch_' + str(i) + '_step_' +str(j) )
                with open(f"~/Documents/project/History/historys{j}e{i}.txt") as f: 
                    for key, value in hist.history.items(): 
                        f.write(f'epochs {i} :\n Key:  {key}  Value:  {value}')
                    f.write('\n####################################################')
            self.net.save(saveAddr+'model_' + str(i) + '.h5')
    @staticmethod
    def plotHist(Hist,saveAddr,i = ''):
        from matplotlib import pyplot as plt
        plt.plot(Hist.history['accuracy'])
        plt.plot(Hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.savefig(saveAddr +'accuracy_'+i+'.png')
        plt.plot(Hist.history['loss'])
        plt.plot(Hist.history['val_loss'])
        plt.title('model loss')
        plt.savefig(saveAddr + 'loss_'+i+'.png')
    def callBacks(self,epoch):
        self.clbk = []
        clbk = tf.keras.callbacks.ModelCheckpoint(filepath = r'~/Documents/projects/SHHS/callBacks/modelCheckpoint',
                                                        monitor = 'val_accuracy',
                                                        mode='max',
                                                        save_best_only=True)
        self.clbk.append(clbk)
        
        clbk = tf.keras.callbacks.TensorBoard(log_dir = r'~/Documents/projects/SHHS/callBacks/tensorboard'+'/'+ epoch +'_log')
        self.clbk.append(clbk)
        
        clbk = tf.keras.callbacks.EarlyStopping()
        self.clbk.append(clbk)
        
    def loadModel(self,addr):
        self.net = tf.keras.models.load_model(addr)
        
