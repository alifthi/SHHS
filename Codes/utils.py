'''
author: alifthi

Email: alifathi8008@gmail.com

lastEdit: 11/aban/01
           2022/NOV/2
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
        self.targetSignals = ['sao2','hr','eeg','eegsec','ecg','emg',
                            'eogl','eogr','thorres','abdores','newair',
                            'light','position']
        self.namesFilter = '.,_ ()'
    def readAsDF(self,signalPath,save = False,id=None,returnOneSignal = False):         # it reads signals as a CSV file
        file = edf.EdfReader(signalPath)                    
        headers = file.getSignalHeaders()  
        if returnOneSignal:                     
            table = pd.DataFrame()
        else:
            List = []
        for i in range(len(headers)):
            signal = list(file.readSignal(i))
            name = headers[i]['label']
            for c in self.namesFilter:
                name = name.replace(c,'')
            name = name.lower()
            if not(name in self.targetSignals):
                continue
            rate = int(headers[i]['sample_rate'])
            chunks = [signal[x:x+rate] for x in range(0, len(signal), rate)]                # this line of code will chunk the main signal in term of sampling frequency
            if returnOneSignal:                     
                table[name] = chunks
            else:
                List.append(chunks)
        if save:
            table.to_csv(self.path2save+'/'+str(id)+'.csv')                                 # saving every signal as csv file 
        file.close()
        if returnOneSignal and len(table.columns) == len(self.targetSignals):                     
            return table
        elif (not returnOneSignal) and len(List) == len(self.targetSignals):
            return List
        else:
            return None
    def globForOnEdfs(self):
        import glob
        normalData = pd.DataFrame(columns=self.targetSignals)
        patientData = pd.DataFrame(columns=self.targetSignals)
        normalCounter = 0
        patientCounter = 0
        for Dir in self.signalDir:
            for path in glob.glob(Dir+'/*'):                                                   # for on every signal in Dir path
                id = int(path.split('/')[-1].split('-')[1].split('.')[0])                      # extract id in the path
                if id in list(self.validNormalSignalsName['id']):                           # condition will be True, if readed ID be our target
                    index = self.validNormalSignalsName.loc[self.validNormalSignalsName['id']==id].index[0]
                    if (len(self.validNormalSignalsName['Signals'][index]) >= len(self.signalQualId)-1):        # condition will be True, if every signal be valid
                        signal = self.readAsDF(signalPath=path)
                        if signal == None:
                            continue
                        normalData.loc[normalCounter] = signal
                        normalCounter +=1   
                        print(normalData.head())                                     
                elif id in list(self.validPatientSignalsName['id']):
                    index = self.validPatientSignalsName.loc[self.validPatientSignalsName['id']==id].index[0]
                    if (len(self.validPatientSignalsName['Signals'][index]) >= (len(self.signalQualId)-1)):
                        signal = self.readAsDF(signalPath=path)
                        if signal == None:
                            continue
                        patientData.loc[patientCounter] = signal
                        patientCounter +=1   
                        print(patientData.head())
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
idPath = '~/Documents/Projects/SHHS/Data/SHHS1.xlsx'
signalQualIdPath = '~/Documents/Projects/SHHS/Data/signal quality.xlsx'
signalQualValuePath = '~/Documents/Projects/SHHS/Data/datasets/1- shhs1-dataset-0.13.0.csv'
path2save = '~/Documents/Projects/SHHS/Data/signalsAsCSV'
signalDir = ['/run/media/alifathi/3e705399-86be-41f0-bf88-34db80a00350/shhs1','/run/media/alifathi/009ba9d2-ab8e-4caa-8e6e-e984f5895971/datapart2/part2']
util = utils(signalDir=signalDir,signalQualIdPath=signalQualIdPath,signalQualValuePath=signalQualValuePath,idPath=idPath,path2save=path2save)
util.readCsv()
util.globForOnEdfs()