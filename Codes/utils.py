'''
author: alifthi

Email: alifathi8008@gmail.com

lastEdit: 10/aban/01
           2022/NOV/11
'''
import pyedflib as edf
import pandas as pd
class utils:
    def __init__(self,signalDir,idPath,signalQualIdPath,signalQualValuePath,path2save):
        self.signalQualIdPath = signalQualIdPath                # saves the path of signal quality id file
        self.signalQualValuePath = signalQualValuePath          # saves the path of signal quality value file
        self.signalDir = signalDir                              # path of signals
        self.idPath = idPath
        self.series = None
        self.path2save = path2save
    def readAsDF(self,signalPath,save = True):                                          # it reads signals as a CSV file
        file = edf.EdfReader(signalPath)
        headers = file.getSignalHeaders()
        table = pd.DataFrame()
        for i in range(len(headers)):
            signal = list(file.readSignal(i))
            name = headers[i]['label']
            rate = int(headers[i]['sample_rate'])
            chunks = [signal[x:x+rate] for x in range(0, len(signal), rate)]
            table[name] = chunks
        if save:
            self.saveAsCSV()
        return table
    def globForOnEdfs(self):
        import glob
        for i in glob.glob(self.signalDir+'*'):
            id = i.split('-')[1].split('.')[0]
            if id in self.validNormalSignalsName['id']:
                index = self.validNormalSignalsName.loc[self.validNormalSignalsName['id']==id].index[0]
                if (len(self.validNormalSignals['Signals'][index]) >= len(self.signalQualId-1)):
                    pass
            elif id in self.validPatientSignalsName['id']:
                index = self.validPatientSignalsName.loc[self.validPatientSignalsName['id']==id].index[0]
                if (len(self.validPatientSignalsName['Signals'][index]) >= (len(self.signalQualId)-1)):
                    pass
    def preprocessing(self):
        pass
    def readCsv(self):                                          # read needed CSV files
        patient = pd.read_excel(self.idPath,sheet_name='Patient')
        absolutelyNormal = pd.read_excel(self.idPath,sheet_name='Absolutely Normal')
        self.signalQualId = ['nsrrid'] +list(pd.read_excel(self.signalQualIdPath)['id'])
        signalQualValue = pd.read_csv(self.signalQualValuePath)
        signalQualColumns = signalQualValue.columns
        signalQualColumns = [i.lower() for i in signalQualColumns]
        signalQualValue.columns = signalQualColumns
        signalQualValue = signalQualValue[self.signalQualId]
        lenPatient = min(len(patient),len(absolutelyNormal))
        lenNormal = lenPatient + 200
        self.validNormalSignalsName = self.isSignalValid(len = lenNormal,list = absolutelyNormal,qualList=signalQualValue,theresh=2,signalQualId=signalQualId)
        self.validPatientSignalsName = self.isSignalValid(len = lenPatient,list = patient,qualList=signalQualValue,theresh=2,signalQualId=signalQualId)
    @staticmethod
    def isSignalValid(len,list,qualList,theresh,signalQualId):      # check that witch signals is valid to use in a EDF file
        goodSignals = pd.DataFrame(columns=['id','Signals'])    
        for i in range(len):
            id = list['nsrrid'][i]
            # qualValue = qualList['nsrrid'][id]
            qualValue = qualList.loc[qualList['nsrrid'] == id].reset_index()
            tmpId = []
            for v  in signalQualId:
                if int(qualValue[v][0])>theresh:
                    tmpId.append(v)
            goodSignals.loc[i] = [id,tmpId]
        return goodSignals
    def saveAsCSV(self):
        self.series.to_csv(self.path2save)    # saving every signal as csv file 
idPath = '/home/alifathi/Documents/Projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '/home/alifathi/Documents/Projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '/home/alifathi/Documents/Projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '/run/media/alifathi/3e705399-86be-41f0-bf88-34db80a00350/CSVSeries/'
util = utils(signalDir=None,signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,idPath=idPath,path2save=path2save)
util.readCsv()