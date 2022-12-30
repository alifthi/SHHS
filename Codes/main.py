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
from model import model as Model
from utils import utils
import numpy as np
import pandas as pd
idPath = '~/Documents/projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['~/Documents/projects/SHHS/Data/part2']
outlayerPath = '~/Documents/projects/SHHS/Data/datasets/outlierCheck.xlsx'
featurePath = '~/Documents/projects/ShhsPhase2/Data/'
allInputNames = ['sao2','hr','eeg','eegsec','ecg','emg','eogl',
                        'eogr','thorres','abdores','newair','light','position']
freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}

inputNames = list(freq.keys())

util = utils(signalDir=signalDir,targetSignals=inputNames,
            signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,
            idPath=idPath,path2save=path2save,outlayerPath=outlayerPath,featurePath=featurePath)

util.readCsv()
[normalData,patientData,patientFeature,normalFeature] = util.globForOnEdfs(normalLen=80,patientLen=80)
patientFeature = patientFeature.drop('index',axis =1 )
normalFeature = normalFeature.drop('index',axis =1 )

valTargets = [0]*(len(normalData))+[1]*(len(patientData))
featureData = pd.concat([normalFeature,patientFeature],axis = 0,ignore_index = True)
featureData = featureData.dropna()
featureData = featureData.reset_index()
featureData = featureData.drop(['index','nsrrid'],axis =1 ).to_numpy()

len(valTargets)
Data = pd.concat([normalData,patientData],axis=0,ignore_index=True)
Data = Data.dropna()
Data = Data.reset_index()
Data = Data.drop('index',axis = 1)
Data.columns = inputNames
print(Data.shape)
del(normalData,patientData)
print('squeeze')
valData = util.squeeze(Data)
valData,valFeatures,valTargets = util.preprocessing(series=valData,targets=valTargets,
                                        features = featureData,
                                        addNoise = False,
                                        split = False)
print('preparing')
[valData,valTargets] = util.prepareData(Data = valData,targets = valTargets)
print('expand')
import pandas as pd
pd.DataFrame(valTargets).to_csv(r'C:\Users\u23\Documents\project\Data\trainData\testTargets.csv')
targetsTest = np.expand_dims(valTargets,axis=-1)
dataTest = []
for i,d in enumerate(valData):
    d = d.reshape([np.shape(d)[1],util.len*util.freq[inputNames[i]]])
    dataTest.append(d)
del(valData,valTargets,d,Data)
print(np.shape(dataTest[0]))
pd.DataFrame(dataTest).to_csv(r'C:\Users\u23\Documents\project\Data\trainData\testData.csv')

from model import model as Model

model = Model(inputNames=inputNames)
model.net = model.buildModel()
model.testData = dataTest.append(featureData)



model.testTargets = targetsTest
saveModelAddr = 'C:\\Users\\u23\\Documents\\project\\Model\\Model\\'
model.compile(saveAddr=saveModelAddr)
model.trainGenerator(util=util,saveAddr = saveModelAddr,split = False)
