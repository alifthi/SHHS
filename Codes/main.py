from model import model 
from utils import utils

idPath = '~/Documents/projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/home/ali/Documents/projects/SHHS/Data/part2']
util = utils(signalDir=signalDir,signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,idPath=idPath,path2save=path2save)
util.readCsv()
[patientSeries,normalSeries,targets] = util.globForOnEdfs()
trainSeries,testSetries,trainTargets,testTarget = util.preprocessing([patientSeries,normalSeries],targets)