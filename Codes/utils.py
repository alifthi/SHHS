import pyedflib as edf
import pandas as pd
class utils:
    def __init__(self,dir):
        self.dir = dir
        self.series = self.readAsDF()
        pass

    def readAsDF(self):
        dir = self.dir
        file = edf.EdfReader(dir)
        headers = file.getSignalHeaders()
        table = pd.DataFrame()
        for i in range(len(headers)):
            signal = list(file.readSignal(i))
            name = headers[i]['label']
            rate = int(headers[i]['sample_rate'])
            chunks = [signal[x:x+rate] for x in range(0, len(signal), rate)]
            table[name] = chunks
        return table
    def preprocessing(self):
        pass