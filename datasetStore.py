import itertools
import pandas as pd

class DataSet:
    """traces and peaks over different ROIs and conditions"""
    def __init__(self, _state="Empty"):
        
        self.DSname = _state
        self.isEmpty = True
    
    def setDSname(self, _name):
        self.DSname = _name
        print("set {}".format(_name))
        
    def getDSname(self):
        print("get {}".format(self.DSname))
        return self.DSname
    
    def addPeaksToDS (self, _resdf):
        # peaks are ResultsDF objects
        # some check ?
        self.resultsDF = _resdf
        self.isempty = False
        
    def addTracesToDS (self, _traces):
        #traceDF object? could just be a dictionary of data frames?
        self.traces = _traces
        self.isempty = False
        print ("addTracesToDS: added")

    def getPeaksFromDS (self, _ROI, _condition):
        pass
        
    def getTracesFromDS (self, _ROI, _condition):
        pass
    
class Store:
    """ a handler for datasets """

    def __init__(self):
        
        self.store = {}
        
    def retrieveWorkingSet(self, name):
        if name in self.store:
            return self.store.pop(name)
        else:
            return None
            
    def storeSet(self, ds):
        #traces should be arranged as ROIs over different conditions
        if ds.DSname in self.store:
            #modify ds.name to allow storage
            ds.name += "_cp"
        
        self.store[ds.DSname] = ds
            
    def listNames(self):
        
        if len(self.store) == 0:
            return []
        else:
            return self.store.keys()
   
        
