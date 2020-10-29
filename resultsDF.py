import itertools
import pandas as pd


class Results:
    """ a data frame for the results """

    def __init__(self, ROI_list=[], condition_list=[], name='All'):
        # the default results table includes all the ROIs.
        
        self.name = name
        self.ROI_list = ROI_list
        self.condition_list = condition_list
        self.extracted = ['t', 'peak']
        
        #print (self.ROI_list, self.condition_list, self.extracted, self.headr)
        #print (type(self.ROI_list), type(self.condition_list), type(self.extracted), type(self.headr))
        if self.ROI_list and self.condition_list:
            self.makeDF()
    
    def makeDF(self):
        self.makeCols()
        self.df = pd.DataFrame([], [0], self.cols)
        #print (self.df.head())
    
    def makeCols(self):
        self.headr = list(itertools.product(self.ROI_list, self.condition_list, self.extracted))
        self.cols = pd.MultiIndex.from_tuples(self.headr)
                
    def addPeaksExtracted (self, _peakDict, _name):
        
        self.name = _name
        self.condition_list = _peakDict.keys()
        
        self.ROI_list = []
        
        for l in _peakDict.values():
            _ROIs = l.columns()
            self.ROI_list = list(set(ROIs+self.ROI_list))   #take only unique ROIs
        
        #what about order?
        print (self.ROI_list)
        self.makeDF()
        for k, v in _peakDict:
            for r in v.columns():
                self.df[r][k] = v[r]
        
        
    
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
