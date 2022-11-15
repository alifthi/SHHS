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

from model import model 
from utils import utils
import numpy as np
import pandas as pd
idPath = '~/Documents/projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/home/ali/Documents/projects/SHHS/Data/part2']
allInputNames = ['sao2','hr','eeg','eegsec','ecg','emg','eogl',
                        'eogr','thorres','abdores','newair','light','position']
freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}
inputNames = list(freq.keys()) # ['ecg']
util = utils(signalDir=signalDir,targetSignals=inputNames,
            signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,
            idPath=idPath,path2save=path2save)
# trainData,testData,trainTargets,testTargets = self.preprocessing(series=Data,targets=targets)

util.readCsv()
[normalData,patientData] = util.globForOnEdfs(normalLen=2,patientLen=2)
[normalData,patientData] = util.globForOnEdfs(normalLen=2,patientLen=2)
targets = [0]*(len(normalData))+[1]*(len(patientData))
print(len(targets))
Data = pd.concat([normalData,patientData],axis=0,ignore_index=True) 
Data = Data.dropna()
Data = Data.reset_index()
Data = Data.drop('index',axis = 1)
Data.columns = inputNames
print(Data)
del(normalData,patientData)
print(len(targets))
Data = util.squeeze(Data,inputNames)
valData,valTargets = util.preprocessing(series=Data,targets=targets,split = False)
targetsTest = np.expand_dims(valTargets,axis=-1)
dataTest = []
for i,d in enumerate(valData):
    d = d.reshape([np.shape(d)[1],1,util.len*util.freq[inputNames[i]]])
    dataTest.append(d)
del(valData,valTargets)
dataGenerator = util.dataGenerator
del(util)
model = model(inputNames=inputNames)
model.compile()
model.trainGenerator(dataGenerator,inputNames=inputNames,valData=[dataTest,targetsTest],
                        saveAddr = '/home/ali/Documents/projects/SHHS/Model/')


# hist = model.trainModel(signal=Data,targets=Targets,validationData=dataTest,validationTargets=targetsTest,batchSize=128,epochs=5)
# model.plotHist(hist)
# model.net.save('/home/ali/Documents/projects/SHHS/Model/modelWithECG30Sec.h5')