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
util.readCsv()
[normalData,patientData] = util.globForOnEdfs(normalLen=7,patientLen=7)
targets = [0]*(len(normalData.T))+[1]*(len(patientData.T))
Data = pd.concat([normalData.T,patientData.T],axis=0,ignore_index=True) 
Data = Data.dropna()
Data = Data.reset_index()
Data = Data.drop('index',axis = 1)
Data.columns = inputNames
print(Data)
del(normalData,patientData)
print(len(targets))
Data = util.squeeze(Data,inputNames)
trainData,testData,trainTargets,testTargets = util.preprocessing(series=Data,targets=targets)
[trainData,trainTargets] = util.prepareData(Data = trainData,targets = trainTargets,inputNames = inputNames)
del(Data,targets)
Targets = np.expand_dims(trainTargets,axis=-1)
Data = []
for i,d in enumerate(trainData):
    d = d.reshape([np.shape(d)[1],1,util.len*util.freq[inputNames[i]]])
    Data.append(d)
[testData,testTargets] = util.prepareData(Data = testData,targets = testTargets,inputNames = inputNames)
targetsTest = np.expand_dims(testTargets,axis=-1)
dataTest = []
for i,d in enumerate(testData):
    d = d.reshape([np.shape(d)[1],1,util.len*util.freq[inputNames[i]]])
    dataTest.append(d)
del(testData)
model = model(inputNames=inputNames)
model.compile()
hist = model.trainModel(signal=Data,targets=Targets,validationData=dataTest,validationTargets=targetsTest,batchSize=128,epochs=5)
model.plotHist(hist)
model.net.save('/home/ali/Documents/projects/SHHS/Model/modelWithECG30Sec.h5')