'''
author: alifthi

Email: alifathi8008@gmail.com

lastEdit: 2022/NOV/5
'''
import pyedflib as edf
import pandas as pd
import numpy as np
class utils:
    def __init__(self,signalDir,idPath,signalQualIdPath,
                signalQualValuePath,path2save,targetSignals):

        self.signalQualIdPath = signalQualIdPath                # saves the path of signal quality id file
        self.signalQualValuePath = signalQualValuePath          # saves the path of signal quality value file
        self.signalDir = signalDir                              # path of signals
        self.idPath = idPath                                    # location that target ids saved
        self.path2save = path2save                              # location to save signal that converted to csv
        self.targetSignals = targetSignals
        self.freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}
        self.len = 32010
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
    def globForOnEdfs(self,normalLen=10,patientLen=10):
        import glob
        normalData = pd.DataFrame(columns=self.targetSignals)
        patientData = pd.DataFrame(columns=self.targetSignals)
        normalCounter = 0
        patientCounter = 0
        for Dir in self.signalDir:
            for path in glob.glob(Dir+'/*'):    
                id = int(path.split('/')[-1].split('-')[1].split('.')[0])                      # extract id in the path
                if id in list(self.validNormalSignalsName['id']):                           	# condition will be True, if readed ID be our target
                    index = self.validNormalSignalsName.loc[self.validNormalSignalsName['id']==id].index[0]
                    if (len(self.validNormalSignalsName['Signals'][index]) >= len(self.signalQualId)-1):        # condition will be True, if every signal be valid
                        if(normalLen == normalCounter):
                            if (patientLen == patientCounter):
                                return [normalData,patientData]
                            else:
                                continue
                        try :
                            signal = self.readAsDF(signalPath=path)
                        except:
                            continue
                        if signal == None:
                            continue 
                        normalData.loc[normalCounter] = signal
                        if normalCounter%5 == 0: 
                            print(f'{normalCounter}th normal signal read!')

                        normalCounter +=1 
                elif id in list(self.validPatientSignalsName['id']):
                    if  (patientLen == patientCounter):
                        if (normalLen == normalCounter) :
                            return [normalData,patientData] 
                        else :
                            continue
                    index = self.validPatientSignalsName.loc[self.validPatientSignalsName['id']==id].index[0]
                    if (len(self.validPatientSignalsName['Signals'][index]) >= (len(self.signalQualId)-1)):
                        signal = self.readAsDF(signalPath=path)
                        if signal == None:
                            continue
                        if patientCounter%20 == 0: 
                            print(f'{patientCounter}th patient signal read!')

                        patientData.loc[patientCounter] = signal
                        patientCounter +=1   
        return [normalData,patientData]
    def prepareData(self,Data,targets,inputNames):
        Data = np.squeeze(Data)
        data =np.expand_dims([Data[0]],axis = 0)
        t = targets[0]
        Targets = np.expand_dims([t],axis = -1)
        for i,s in enumerate(Data[1:]):
            index = self.len*self.freq[inputNames[0]]
            if len(s)>index:
                s = s[:index]    
            else :
                s += [0]*(index-len(s))
            tmp = np.expand_dims([s],axis = 0)    
            t = targets[i]
            t = np.expand_dims([t],axis = -1)
            data = np.concatenate([data,tmp],axis=1)
            Targets = np.concatenate([Targets,t],axis=0)
        return [data,Targets]
    @staticmethod
    def preprocessing(series,targets):
        from sklearn.model_selection import train_test_split
        trainSeries,testSetries,trainTargets,testTarget = train_test_split(series,targets,test_size = 0)
        return [trainSeries,testSetries,trainTargets,testTarget]
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
        lenNormal = lenPatient
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
    @staticmethod
    def squeeze(Data,names):
        data = []
        for j in range(len(Data)):
            array = []
            for i in names:   
                tmpArray = np.asarray(Data[i][j])
                shape = tmpArray.shape
                tmpArray = np.reshape(tmpArray,[shape[0]*shape[1]])
                tmpArray = np.asarray(tmpArray).astype('float32').tolist()            
                array.append(tmpArray)
            data.append(array)
        return data
    def buildTimeseriesGenerator(self,series,targets):
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
        gen = TimeseriesGenerator(series,targets,length = self.len,sampling_rate = self.freq.values)
        return gen