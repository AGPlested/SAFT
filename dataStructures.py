import itertools
import pandas as pd
import numpy as np
from utils import maskPeaks

def histogramFitParams(conditions, pColumns=None):
    if pColumns == None:
        
        #default columns
        pColumns = ['ROI', 'id', 'N', 'Pr/mu', 'Stat', 'statistic', 'P', 'type']
    
    _a = [conditions, pColumns]
    
    _cols =pd.MultiIndex.from_product(_a, names=('conditions', 'fit params'))
    return pd.DataFrame(columns=_cols)     # dataframe for Pr values from fitting

class HistogramFitStore:
    def __init__(self, _ROI, _conditions):
        self.ROI = _ROI
        self.conditions = list(_conditions)
        _c = [self.conditions, ["Hx", "Hy", "Fitx", "Fity"]]
        _cols =pd.MultiIndex.from_product(_c, names=("conditions", "data"))
        self.df = pd.DataFrame (columns=_cols)
        self.empty = True
        #print ("df.head \n{}".format(self.df.head(5)))
    
    def addHData(self, condition, hx, hy):
        _c = condition
        sHx = pd.Series(hx)
        sHy = pd.Series(hy)
        
        # both indices *should* be monotonic ascending integer
        if len (self.df.index) < len (sHx.index):
            self.df = self.df.reindex(sHx.index)
        
        self.df.loc(axis=1)[(_c, "Hx")] = sHx
        self.df.loc(axis=1)[(_c, "Hy")] = sHy
        self.empty = False
    
    def addFData(self, condition, fitx, fity):
        _c = condition
        sFx = pd.Series(fitx)
        sFy = pd.Series(fity)
        
        # both indices *should* be monotonic ascending integer
        if len (self.df.index) < len (sFx.index):
            self.df = self.df.reindex(sFx.index)
        
        self.df.loc(axis=1)[(_c, "Fitx")] = sFx
        self.df.loc(axis=1)[(_c, "Fity")] = sFy
        self.empty = False
        
    def addHFData(self, condition, hx, hy, fitx, fity):
    
        self.addHData(condition, hx, hy)
        self.addFData(condition, fitx, fity)
        
    
        

class HistogramsR():
    """ a data frame for the common histogram result """

    def __init__(self, ROI_list, set_list, Nbins, binStart, binEnd):
        #the default histogram table includes all the ROIs.
        #note, the bin edges are stored outside the dataframe (Nbins + 1)
        
        self.ROI_list = ROI_list
        self.set_list = set_list
        
        _, self.binEdges = np.histogram([0], bins = Nbins, range = (binStart, binEnd))
        self.headr = list(itertools.product(self.ROI_list, self.set_list))
        #print (self.ROI_list, self.set_list, self.extracted, self.headr)
        print (self.binEdges)
        self.cols = pd.MultiIndex.from_tuples(self.headr)
        
        self.df = pd.DataFrame([], range(Nbins), self.cols)
        print (self.df.head())
        
    def ROI_sum (self):
        
        for _ROI in list(self.df.columns.levels[0]):
            self.addHist (_ROI, "Sum", self.df[_ROI].sum(axis=1))
        
    def addHist (self, _ROI, _condition, _h):
        # the histogram are arrays of values that belong to a ROI and a set (condition) or a sum for the ROI.
        #_h = np.append(_h, [np.nan])       #to equalize lengths
        
        print (_ROI, _condition, _h)
        print (len(self.df.index), len(_h))
        # overwrite or add column if new
        self.df[_ROI, _condition] = pd.Series(_h)
        print (self.df[_ROI][_condition])
        

    def getHist (self, _ROI, _condition):
        
        _hx = self.binEdges
        _hy = self.df[_ROI, _condition]
        
        return _hy, _hx #to match np.histogram


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
        #print (self.df[_ROI, _condition, 'peak'])


    def getPeaks (self, _ROI, _condition):
        # all columns except the longest end in NaN
        # "empty" columns are just NaN
        _times = self.df[_ROI, _condition, 't'].dropna()
        _peaks = self.df[_ROI, _condition, 'peak'].dropna()
        
        return _times, _peaks



class Dataset:
    """ named collection of traces, peaks and associated GUI controls over different ROIs and conditions"""
    def __init__(self, _state="Empty"):
        
        self.DSname = _state
        self.isEmpty = True
        self.GUIcontrols = {}
        self.GUIcontrols["autoPeaks"] = "Enable"   # a dataset can activate/deactivate parts of the GUI, activated by default
        self.ROI_list = []
        self.trace = None
        self.peakTimes = pd.Series([])
    
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

    def getSD (self, maskWidth=10):
        if self.isempty:
            return None
        SD = {}
        if self.traces:
            
            # need to get the peaks
            # get SD should only be called after peaks were found - should be an option
            peakTimes = self.peakTimes.values
            print ("peakTimes", peakTimes)
            for i, condition in enumerate(self.traces):
                _stc = self.traces[condition]
                # find the row indices in the trace dataframe that match the times of the peaks
                peaksIdx = _stc[_stc.index.isin(peakTimes)].index.values
                print ("Peaks idx", peaksIdx)
                peaksOmitted = maskPeaks (_stc, peaksIdx, maskWidth)
                
                SD[i] = peaksOmitted.std()
                old = _stc.std()
                print ("old, masked SDi: {} {}".format(old, SD[i]))
                idx = SD[i].index   # all the same so use the last one below
                
            allSD = np.vstack([s.transpose() for s in SD.values()])
            
            allSD_df = pd.DataFrame(allSD, columns=idx)
            print (allSD_df, allSD_df.min())
            return allSD_df.min()
        
        else:
            return None
        
 
 
class Store:
    """ a dictionary handler for storing datasets """

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
            ds.DSname += "_cp"
        
        self.store[ds.DSname] = ds
            
    def listNames(self):
        
        if len(self.store) == 0:
            return []
        else:
            return self.store.keys()
   
        
