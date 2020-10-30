import itertools
import pandas as pd

class Results:
    """ a custom data frame for the peak results """

    def __init__(self, ROI_list=[], condition_list=[], name='All'):
        # the default results table includes all the ROIs.
        
        self.name = name
        self.ROI_list = ROI_list
        self.condition_list = condition_list
        self.pairs = ['t', 'peak']
        
        #print (self.ROI_list, self.condition_list, self.pairs, self.headr)
        #print (type(self.ROI_list), type(self.condition_list), type(self.pairs), type(self.headr))
        if self.ROI_list and self.condition_list:
            self.makeDF()
    
    def makeDF(self, _index=[0]):
        self.makeCols()
        self.df = pd.DataFrame([], _index, self.cols)
        #print (self.df.head())
    
    def makeCols(self):
        self.headr = list(itertools.product(self.ROI_list, self.condition_list, self.pairs))
        self.cols = pd.MultiIndex.from_tuples(self.headr)
                
    def addPeaksExtracted (self, _peakDict, _name=None):
        for d in _peakDict.values():
            print ("dhead\n\n{}".format(d.head(5)))
        if _name:
            self.name = _name
        
        self.condition_list = list(_peakDict.keys())
        
        # reset in case
        self.ROI_list = []
        for df in _peakDict.values():
            _ROIs = df.columns
            self.ROI_list = list(set(list(_ROIs) + list(self.ROI_list)))   #take only unique ROIs
        
        #what about order?
        
        # in extracted data they should be all the same.
        # could check that
        _index=_peakDict[self.condition_list[0]].index
        self.makeDF(_index)
        
        rs = self.df.columns.get_level_values(0).unique()
        condi = self.df.columns.get_level_values(1).unique()

        for rx in rs:
            for c , d  in _peakDict.items():
                self.df.loc(axis=1)[rx, c, 't'] = _index.to_series().values
                self.df.loc(axis=1)[rx, c, 'peak'] = d[rx]
        
        self.df.reset_index(drop=True, inplace=True)
        
        print ("selfdfhead\n\n{}".format(self.df.head(5)))
    
    def addPeaks (self, _ROI, _condition, _times, _peaks):
        # the peaks (and their times) are arrays of values that belong to a ROI and a condition.
        
        # list of peaks will be of arbitrary length
        # check that it is not too long for the dataFrame
        print ("addPeaks: self.df.index.size, lenpeaks:", self.df.index.size, len(_peaks) )
        if self.df.index.size < len (_peaks):
            _rlp = range(len(_peaks))
            self.df = self.df.reindex(_rlp)
                
        # overwrite or add column if new
        self.df[_ROI, _condition, 't'] = pd.Series(_times)
        self.df[_ROI, _condition, 'peak'] = pd.Series(_peaks)
        print (self.df[_ROI, _condition, 'peak'])


    def getPeaks (self, _ROI, _condition):
        # all columns except the longest end in NaN
        # "empty" columns are just NaN
        _times = self.df[_ROI, _condition, 't'].dropna()
        _peaks = self.df[_ROI, _condition, 'peak'].dropna()
        
        return _times, _peaks





class Dataset:
    """traces and peaks over different ROIs and conditions"""
    def __init__(self, _state="Empty"):
        
        self.DSname = _state
        self.isEmpty = True
        self.GUIcontrols = {}
        self.GUIcontrols["autopeaks"] = 'Enable'   # a dataset can activate/deactivate parts of the GUI
    
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
   
        
