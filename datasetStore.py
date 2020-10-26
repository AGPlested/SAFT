import itertools
import pandas as pd

class DataSet():
    """traces and peaks over different ROIs and conditions"""
    def __init__(self):
        
        self.name = "unnamed"
        self.isempty = True
    
    def setSetName(self, _name):
        self.name = _name
    
    
    
    def addPeaksToSet(self, _resdf):
        # peaks are ResultsDF objects
        # some check ?
        self.resultsDF = _resdf
        self.isempty = False
        
    def addTracesToSet (self, _traces):
        #traceDF object? could just be a dictionary of data frames?
        self.traces = _traces
        self.isempty = False

    def getPeaksFromSet (self, _ROI, _condition):
        pass
        
    def getTracesFomSet (self, _ROI, _condition):
        pass
    
class Store():
    """ a handler for datasets """

    def __init__(self):
        
        self.store = {}
        
    def retrieveWorkingSet(self, name):
        if name in store:
            return store.pop(name)
        else:
            return None
            
    def storeSet(self, ds):
        #traces should be arranged as ROIs over different conditions
        if ds.name in self.store:
            #modify ds.name to allow storage
            ds.name += "_cp"
        
        self.store[ds.name] = ds
            
    def listNames(self):
        
        if len(self.store) == 0:
            return []
        else:
            return self.store.keys()
   
        
