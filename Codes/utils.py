'''
The MIT License (MIT)

Copyright (c) 2022 Ali Fathi Jahromi <alifathi8008@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import pyedflib as edf
import pandas as pd
import numpy as np
class utils:
    def __init__(self,signalDir,idPath,signalQualIdPath,
                signalQualValuePath,path2save,targetSignals,outlayerPath):
        
        self.outlayerPath = outlayerPath
        self.signalQualIdPath = signalQualIdPath                # saves the path of signal quality id file
        self.signalQualValuePath = signalQualValuePath          # saves the path of signal quality value file
        self.signalDir = signalDir                              # path of signals
        self.idPath = idPath                                    # location that target ids saved
        self.path2save = path2save                              # location to save signal that converted to csv
        self.targetSignals = targetSignals
        self.freq = {'sao2': 1, 'hr': 1, 'eogl': 50 ,'eogr': 50, 'eeg': 125,'eegsec': 125,'ecg': 125,
                    'emg': 125, 'thorres': 10, 'abdores': 10, 'position': 1, 'light': 1, 'newair': 10}
        self.len = 120 # 32010
        self.namesFilter = '.,_ ()'
        self.readSignals  = []                  #  this list contain data that read
        self.invalidSignals = []
    def readAsDF(self,signalPath,save = False,id=None,returnOneSignal = False):         # it reads signals as a CSV file
        file = edf.EdfReader(signalPath)                    
        headers = file.getSignalHeaders() 
        names =[]
        if returnOneSignal:                     
            table = pd.DataFrame()
        else:
            List = []
        for i in range(len(headers)):
            name = headers[i]['label']
            for c in self.namesFilter:
                name = name.replace(c,'')
            name = name.lower()
            if not(name in self.targetSignals):
                continue
            names.append(name)   
        if not(len(names) == len(self.targetSignals)):
            return None
        names = []
        for name in headers:
            names.append(name['label'])
        for i,name in enumerate(names):
            for c in self.namesFilter:
                name = name.replace(c,'')
                name = name.lower()
            names[i] = name
        for name in self.targetSignals: # range(len(names)):

            if not(name in names):
                return None
            i = names.index(name)
            rate = int(headers[i]['sample_rate'])
            lowerRate = self.len
            nHoure = 3
            signal = list(file.readSignal(i))
            chunks = [signal[x:x+rate] for x in range(0, rate*nHoure*3600, rate)]                # this line of code will chunk the main signal in term of sampling frequency
            lowerChunk = [chunks[x:x+lowerRate] for x in range(0, len(chunks), lowerRate)]
            if returnOneSignal:  
                table[name] = lowerChunk
            else:
                List.append(lowerChunk)
        if save:
            table.to_csv(self.path2save+'\\'+str(id)+'.csv')                                 # saving every signal as csv file 
        file.close()
        if returnOneSignal: # and len(table.columns) == len(self.targetSignals):                     
            return table
        elif (not returnOneSignal) and len(List) == len(self.targetSignals):
            return List
        else:
            return None
        
    def globForOnEdfs(self,normalLen=10,patientLen=10):
        import glob
        normalLen -=1
        patientLen -=1
        normalData = []
        patientData = []
        normalCounter = 0
        patientCounter = 0
        for Dir in self.signalDir:
            for path in glob.glob(Dir+'\\*'):
                id = int(path.split('\\')[-1].split('-')[1].split('.')[0])            # extract id in the path
                if id in self.invalidSignals:
                    continue
                if id in self.readSignals:
                    continue
                if id in list(self.validNormalSignalsName['id']):                           	# condition will be True, if readed ID be our target
                    index = self.validNormalSignalsName.loc[self.validNormalSignalsName['id']==id].index[0]
                    if (len(self.validNormalSignalsName['Signals'][index]) >= len(self.signalQualId)-1):        # condition will be True, if every signal be valid
                        if(normalLen == normalCounter):
                            if (patientLen == patientCounter):
                                normalData = pd.concat(normalData,axis=0,ignore_index=True)
                                patientData = pd.concat(patientData,axis=0,ignore_index=True)
                                return [normalData,patientData]
                            else:
                                continue
                        try :
                            signal = self.readAsDF(signalPath=path,id = id,returnOneSignal=True) 
                        except:
                            continue
                        try:
                            if signal == None:
                                if not (id in self.invalidSignals):
                                       self.invalidSignals.append(id)
                                continue
                        except:
                            pass
                        if normalCounter%10 == 0:
                            print(f'normal: {normalCounter}')
                        self.readSignals.append(id)
                        normalData.append(signal)
                        normalCounter +=1 
                elif id in list(self.validPatientSignalsName['id']):
                    if  (patientLen == patientCounter):
                        if (normalLen == normalCounter) :
                            normalData = pd.concat(normalData,axis=0,ignore_index=True)
                            patientData = pd.concat(patientData,axis=0,ignore_index=True)
                            return [normalData,patientData] 
                        else :
                            continue
                    index = self.validPatientSignalsName.loc[self.validPatientSignalsName['id']==id].index[0]
                    if (len(self.validPatientSignalsName['Signals'][index]) >= (len(self.signalQualId)-1)):
                        signal = self.readAsDF(signalPath=path,returnOneSignal=True)
                        try:
                            if signal == None:
                                if not (id in self.invalidSignals):
                                       self.invalidSignals.append(id)
                                       
                                continue
                        except:
                            pass
                        if patientCounter%10 == 0:
                            print(f'patient: {patientCounter}')
                        self.readSignals.append(id)
                        patientData.append(signal)
                        patientCounter +=1
        try:
            normalData = pd.concat(normalData,axis=0,ignore_index=True)
            patientData = pd.concat(patientData,axis=0,ignore_index=True)
            normalData = normalData.T
            patientData = patientData.T
            return [normalData,patientData]
        except:
            return None
    def prepareData(self,Data,targets):
        mainData = []
        tar = []
        for n,name in enumerate(self.targetSignals):
            data = [] 
            for i,s in enumerate(Data):
                s = s[n]
                index = self.len*self.freq[name]
                if len(s)<index:
                    continue

                tmp = np.expand_dims([s],axis = 0)   
                data.append(tmp)
                if n == 0 : 
                    t = targets[i]
                    t = np.expand_dims([t],axis = -1) 
                    tar.append(t)
            Targets = np.concatenate(tar,axis=0)
            data = np.concatenate(data,axis=1)
            mainData.append(data)
        return [mainData,Targets]
    def preprocessing(self,series,targets,addNoise = False,split = True):
        for i in range(len(series)):
            for j in range(len(self.targetSignals)):
                m = max(max(series[i][j]),1)
                for z in range(len(series[i][j])):
                    series[i][j][z] /= m
        if split:
            from sklearn.model_selection import train_test_split
            trainSeries,testSetries,trainTargets,testTarget = train_test_split(series,targets,test_size = 0.1)
            if addNoise:            
                trainSeries = trainSeries + np.random.normal(0,1,np.shape(trainSeries))
            return [trainSeries,testSetries,trainTargets,testTarget]
        return [series,targets]
    def readCsv(self):                                          # read needed CSV files
        patient = pd.read_excel(self.idPath,sheet_name='Patient')
        absolutelyNormal = pd.read_excel(self.idPath,sheet_name='Absolutely Normal')
        self.signalQualId = ['nsrrid'] + list(pd.read_excel(self.signalQualIdPath)['id'])
        self.outlayerID =  ['nsrrid'] + list(pd.read_excel(self.outlayerPath)['id'])
        signalQualValue = pd.read_csv(self.signalQualValuePath)
        signalQualColumns = signalQualValue.columns
        signalQualColumns = [i.lower() for i in signalQualColumns]
        signalQualValue.columns = signalQualColumns
        outlayerValue = signalQualValue[self.outlayerID]
        signalQualValue = signalQualValue[self.signalQualId]
        lenPatient = min(len(patient),len(absolutelyNormal))
        lenNormal = lenPatient
        self.validNormalSignalsName = self.isSignalValid(Len = lenNormal,
                                                         list = absolutelyNormal,
                                                         qualList=signalQualValue,
                                                         thereshQual=2,
                                                         signalQualId=self.signalQualId,
                                                         outlayerValue = outlayerValue,
                                                         outlayerID = self.outlayerID)
        
        self.validPatientSignalsName = self.isSignalValid(Len = lenPatient,
                                                          list = patient,
                                                          qualList=signalQualValue,
                                                          thereshQual=2,
                                                          signalQualId=self.signalQualId,
                                                          outlayerValue = outlayerValue,
                                                          outlayerID = self.outlayerID)
    @staticmethod
    def isSignalValid(Len,list,qualList,thereshQual,signalQualId,outlayerValue,outlayerID):      # check that witch signals is valid to use in a EDF file
        goodSignals = pd.DataFrame(columns=['id','Signals'])    
        for i in range(Len):
            id = list['nsrrid'][i]
            qualValue = qualList.loc[qualList['nsrrid'] == id].reset_index()
            outlayer = outlayerValue.loc[outlayerValue['nsrrid'] == id].fillna(1).reset_index()
            tmpId = []
            for v  in signalQualId:
                if (int(qualValue[v][0])>thereshQual):
                    if ((int(outlayer[outlayerID[1]][0]) == 0 )and (int(outlayer[outlayerID[2]][0]) == 0) ) and (int(outlayer[outlayerID[3]][0]) >3):
                        tmpId.append(v)

            goodSignals.loc[i] = [int(id),tmpId]
        return goodSignals
    def squeeze(self,Data):
        data = []
        for j in range(len(Data)):
            array = []
            for i in self.targetSignals :   
                shape = np.shape(Data[i][j])
                tmpArray = np.reshape(Data[i][j],[shape[1]*shape[0],1])
                array.append(tmpArray)
            data.append(array)
        return data
    def dataGenerator(self,normalLen,patientLen):
        
        self.readCsv()
        [normalData,patientData] = self.globForOnEdfs(normalLen=normalLen,patientLen=patientLen)
        print('preparing data...')
        if len(normalData) <= 15:
            normalData = normalData.T
            patientData = patientData.T
        trainTargets = [0]*(len(normalData))+[1]*(len(patientData))
        Data = pd.concat([normalData,patientData],axis=0,ignore_index=True) 
        Data = Data.dropna()
        Data = Data.reset_index()
        Data = Data.drop('index',axis = 1)
        Data.columns = self.targetSignals
        del(normalData,patientData)
        trainData = self.squeeze(Data)
        del(Data)
        trainData,testData,trainTargets,testTargets = self.preprocessing(series=trainData,targets=trainTargets,split = True)
        [trainData,trainTargets] = self.prepareData(Data = trainData,targets = trainTargets)
        trainT = np.squeeze(trainTargets)
        trainD = []
        for i,d in enumerate(trainData):
            d = d.reshape([np.shape(d)[1],self.len*self.freq[self.targetSignals[i]]])
            trainD.append(d)
        del(trainData,trainTargets)
        [testData,trainTargets] = self.prepareData(Data = testData,targets = testTargets)
        testT = np.squeeze(trainTargets)
        testD = []
        for i,d in enumerate(testData):
            d = d.reshape([np.shape(d)[1],self.len*self.freq[self.targetSignals[i]]])
            testD.append(d) 
        return [trainD,trainT,testD,testT]
