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
[normalData,patientData] = util.globForOnEdfs(normalLen=2,patientLen=2)
targets = [0]*(len(normalData))+[1]*(len(patientData))  # T
Data = pd.concat([normalData,patientData],axis=0,ignore_index=True) # T and 0
Data = Data.dropna()
Data = Data.reset_index()
Data = Data.drop('index',axis = 1)
print(Data.head())
del(normalData,patientData)
print(Data.shape)
Data = util.squeeze(Data,util.targetSignals)
[data,Targets] = util.prepareData(Data = Data,targets = targets,inputNames = util.targetSignals)
del(Data,targets)
model = model(inputNames=util.targetSignals)
model.compile()
Targets = np.expand_dims(Targets,axis=-1)
Data = []
for i,d in enumerate(data):
    d = d.reshape([np.shape(d)[1],1,util.len*util.freq[util.targetSignals[i]]])
    Data.append(d)
del(data)
del(util)
model.trainModel(signal=Data,targets=Targets,batchSize=512,epochs=3)
model.net.save('/home/ali/Documents/projects/SHHS/Model/modelWithECG.h5')