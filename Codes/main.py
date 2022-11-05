from model import model 
from utils import utils
import pandas as pd
idPath = '~/Documents/projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/home/ali/Documents/projects/SHHS/Data/part2']
util = utils(signalDir=signalDir,signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,idPath=idPath,path2save=path2save)
util.readCsv()
series = util.globForOnEdfs()
targets = [0]*len(series[0])+[1]*len(series[1])
df = pd.concat(series).reset_index()
model = model()
