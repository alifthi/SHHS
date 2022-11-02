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
        self.idPath = idPath                                    # location that target ids saved
        self.path2save = path2save                              # location to save signal that converted to csv
    def readAsDF(self,signalPath,save = False,id=None):         # it reads signals as a CSV file
        file = edf.EdfReader(signalPath)                    
        headers = file.getSignalHeaders()                       
        table = pd.DataFrame()
        for i in range(len(headers)):
            signal = list(file.readSignal(i))
            name = headers[i]['label']
            rate = int(headers[i]['sample_rate'])
            chunks = [signal[x:x+rate] for x in range(0, len(signal), rate)]                # this line of code will chunk the main signal in term of sampling frequency
            table[name] = chunks
        if save:
            table.to_csv(self.path2save+'/'+str(id)+'.csv')                                 # saving every signal as csv file 
        file.close()
        return table
    def globForOnEdfs(self):
        import glob
        for Dir in self.signalDir:
            for i in glob.glob(Dir+'/*'):                                                   # for on every signal in Dir path
                id = int(i.split('/')[-1].split('-')[1].split('.')[0])                      # extract id in the path
                if id in list(self.validNormalSignalsName['id']):                           # condition will be True, if readed ID be our target
                    index = self.validNormalSignalsName.loc[self.validNormalSignalsName['id']==id].index[0]
                    if (len(self.validNormalSignalsName['Signals'][index]) >= len(self.signalQualId)-1):        # condition will be True, if every signal be valid
                        self.readAsDF(signalPath=i)                                        
                elif id in list(self.validPatientSignalsName['id']):
                    index = self.validPatientSignalsName.loc[self.validPatientSignalsName['id']==id].index[0]
                    if (len(self.validPatientSignalsName['Signals'][index]) >= (len(self.signalQualId)-1)):
                        self.readAsDF(signalPath=i)
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
        self.validNormalSignalsName = self.isSignalValid(len = lenNormal,list = absolutelyNormal,qualList=signalQualValue,theresh=2,signalQualId=self.signalQualId)
        self.validPatientSignalsName = self.isSignalValid(len = lenPatient,list = patient,qualList=signalQualValue,theresh=2,signalQualId=self.signalQualId)
    @staticmethod
    def isSignalValid(len,list,qualList,theresh,signalQualId):      # check that witch signals is valid to use in a EDF file
        goodSignals = pd.DataFrame(columns=['id','Signals'])    
        for i in range(len):
            id = list['nsrrid'][i]
            qualValue = qualList.loc[qualList['nsrrid'] == id].reset_index()
            tmpId = []
            for v  in signalQualId:
                if int(qualValue[v][0])>theresh:
                    tmpId.append(v)
            goodSignals.loc[i] = [int(id),tmpId]
        return goodSignals
idPath = '/home/alifathi/Documents/Projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '/home/alifathi/Documents/Projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '/home/alifathi/Documents/Projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '/home/alifathi/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/run/media/alifathi/3e705399-86be-41f0-bf88-34db80a00350/shhs1','/run/media/alifathi/009ba9d2-ab8e-4caa-8e6e-e984f5895971/datapart2/part2']
util = utils(signalDir=signalDir,signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,idPath=idPath,path2save=path2save)
util.readCsv()
util.globForOnEdfs()