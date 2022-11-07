from model import model 
from utils import utils
import numpy as np
import pandas as pd
import tensorflow as tf
idPath = '~/Documents/projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/home/ali/Documents/projects/SHHS/Data/part2']
allInputNames = ['sao2','hr','eeg','eegsec','ecg','emg','eogl',
                        'eogr','thorres','abdores','newair','light','position']
inputNames = ['ecg'] #,'eegsec','eogl','eogr']
util = utils(signalDir=signalDir,targetSignals=inputNames,
            signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,
            idPath=idPath,path2save=path2save)
util.readCsv()
[normalData,patientData] = util.globForOnEdfs()
targets = [0]*len(normalData)+[1]*len(patientData)
Data = pd.concat([normalData,patientData]).reset_index()
Data = util.squeeze(Data,inputNames)
Data = np.squeeze(Data)
# Data = np.expand_dims(Data,axis=-1)
# r = np.squeeze(b,axis=-1)


# t = targets[0],r = np.expand_dims([t],axis = -1), e = np.expand_dims([d[0]],axis = 0),d = np.squeeze(Data)
# d = np.squeeze(Data).tolist()
# d = np.expand_dims(d,axis = -1)

data =np.expand_dims([Data[0]],axis = 0)
t = targets[1]
Targets = np.expand_dims([t],axis = -1)
for i,s in enumerate(Data[1:]):
    index = util.len*util.freq[inputNames[0]]
    if len(s)>index:
        s = s[:index]    
    else :
        s += [0]*(index-len(s))
    tmp = np.expand_dims([s],axis = 0)    
    t = targets[i]
    t = np.expand_dims([t],axis = -1)
    data = np.append(data,tmp,axis=1)
    Targets = np.append(Targets,t,axis=1)
# targets  = np.expand_dims(targets, 0)
print(type(Data))
del(normalData,patientData)
model = model(inputNames=inputNames)
model.compile()
print(np.shape(Data))
model.trainModel(signal=data,targets=Targets,batchSize=5)