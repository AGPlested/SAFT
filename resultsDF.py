import itertools
import pandas as pd

class Results():
    """ a data frame for the results """

    def __init__(self, ROI_list, set_list, name='All'):
        #the default results table includes all the ROIs.
        
        self.name = name
        self.ROI_list = ROI_list
        self.set_list = set_list
        self.extracted = ['t', 'peak']
        
        self.headr = list(itertools.product(self.ROI_list, self.set_list, self.extracted))
        #print (self.ROI_list, self.set_list, self.extracted, self.headr)
        #print (type(self.ROI_list), type(self.set_list), type(self.extracted), type(self.headr))
        self.cols = pd.MultiIndex.from_tuples(self.headr)
        
        self.df = pd.DataFrame([], [0], self.cols)
        print (self.df.head())
    
    
    
    def addPeaks (self, _ROI, _set, _times, _peaks):
        # the peaks (and their times) are arrays of values that belong to a ROI and a set (condition).
        
        # list of peaks will be of arbitrary length
        # check that it is not too long for the dataFrame
        print ("addPeaks: self.df.index.size, lenpeaks:", self.df.index.size, len(_peaks) )
        if self.df.index.size < len (_peaks):
            _rlp = range(len(_peaks))
            self.df = self.df.reindex(_rlp)
                
        # overwrite or add column if new
        self.df[_ROI, _set, 't'] = pd.Series(_times)
        self.df[_ROI, _set, 'peak'] = pd.Series(_peaks)
        print (self.df[_ROI, _set, 'peak'])

    def getPeaks (self, _ROI, _set):
        # all columns except the longest end in NaN
        # "empty" columns are just NaN
        _times = self.df[_ROI, _set, 't'].dropna()
        _peaks = self.df[_ROI, _set, 'peak'].dropna()
        
        return _times, _peaks

    def readInPeakDialogResults(self, gpdResults):
        
        print ("readInPeakDialogResults is not used any more")
        
