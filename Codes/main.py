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
idPath = 'C:\\Users\\u23\\Documents\\project\\SHHS\\Data\\SHHS1.xlsx'
signalQualIdPath = 'C:\\Users\\u23\\Documents\\project\\SHHS\Data\\signal quality.xlsx'
signalQualValuePath = 'C:\\Users\\u23\\Documents\\project\\SHHS\\Data\\datasets\\1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['G:\\projects\\shhs\\shhs1\\edf']
allInputNames = ['sao2','hr','eeg','eegsec','ecg','emg','eogl',
                        'eogr','thorres','abdores','newair','light','position']
freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}
inputNames = list(freq.keys()) # ['ecg']
util = utils(signalDir=signalDir,targetSignals=inputNames,
            signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,
            idPath=idPath,path2save=path2save)

util.readCsv()
[normalData,patientData] = util.globForOnEdfs(normalLen=4,patientLen=4)
valTargets = [0]*(len(normalData))+[1]*(len(patientData))
len(valTargets)
Data = pd.concat([normalData,patientData],axis=0,ignore_index=True) 
Data = Data.dropna()
Data = Data.reset_index()
Data = Data.drop('index',axis = 1)
Data.columns = inputNames
print(Data.shape)
del(normalData,patientData)
print('squeeze')
valData = util.squeeze(Data,inputNames)
# valData,valTargets = util.preprocessing(series=valData,targets=targets,split = False)
print('preparing')
[valData,valTargets] = util.prepareData(Data = valData,targets = valTargets,inputNames = inputNames)
print('expand')
targetsTest = np.expand_dims(valTargets,axis=-1)
dataTest = []
for i,d in enumerate(valData):
    d = d.reshape([np.shape(d)[1],1,util.len*util.freq[inputNames[i]]])
    dataTest.append(d)
del(valData,valTargets,d,Data)
print(np.shape(dataTest[0]))
model = model(inputNames=inputNames)
# model.loadModel(addr = r'C:\Users\u23\Documents\project\Model2\model_1_1.h5')
model.compile()
testId = [200077,200116,200081,200082,200093,200114,200115,200117] # util.readSignals
model.trainGenerator(util=util,inputNames=inputNames,valData=[dataTest,targetsTest],
                     saveAddr = 'C:\\Users\\u23\\Documents\\project\\Model2\\')


# hist = model.trainModel(signal=Data,targets=Targets,validationData=dataTest,validationTargets=targetsTest,batchSize=128,epochs=5)
# model.plotHist(hist)
# model.net.save('/home/ali/Documents/projects/SHHS/Model/modelWithECG30Sec.h5')