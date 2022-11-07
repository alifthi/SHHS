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
inputNames = ['ecg']
util = utils(signalDir=signalDir,targetSignals=inputNames,
            signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,
            idPath=idPath,path2save=path2save)
util.readCsv()
[normalData,patientData] = util.globForOnEdfs()
targets = [0]*len(normalData)+[1]*len(patientData)
Data = pd.concat([normalData,patientData]).reset_index()
del(normalData,patientData)
Data = util.squeeze(Data,inputNames)
[data,Targets] = util.prepareData(Data = Data,targets = targets,inputNames = inputNames)
del(Data,targets)
model = model(inputNames=inputNames)
model.compile()
Targets = np.expand_dims(Targets,axis=-1)
data = data.reshape([np.shape(data)[1],1,util.len*util.freq[inputNames[0]]])
del(util)
model.trainModel(signal=data,targets=Targets,batchSize=10,epochs=3)
model.net.save('/home/ali/Documents/projects/SHHS/Model/modelWithECG.h5')