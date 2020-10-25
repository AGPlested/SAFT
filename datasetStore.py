import itertools
import pandas as pd

class DataSet():
    """traces and peaks over different ROIs and conditions"""
    def __init__(self):
        
        self.name = "unnamed"
    
    def setSetName(self, _name):
        self.name = _name
    
    def addPeaksToSet(self, _df):
        # peaks are ResultsDF objects
        # some check that it is a data frame?
        self.peaksDF = _df
        
    def addTracesToSet (self, _traces):
        #traceDF object?
        self.tracesDF = _traces

    def getPeaksFromSet (self, _ROI, _condition):
        
        
    def getTracesFomSet (self, _ROI, _condition):
    
    
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
            
    def giveNames(self):
        
        if len(self.store) == 0:
            return ['-']
        else:
            return self.store.keys()
   
    

    def readInPeakDialogResults(self, gpdResults):
        
        print ("readInPeakDialogResults is not used any more")
        
