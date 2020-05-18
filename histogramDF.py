import itertools
import pandas as pd
import numpy as np

class HistogramsR():
    """ a data frame for the histogram results """

    def __init__(self, ROI_list, set_list, Nbins, binStart, binEnd):
        #the default results table includes all the ROIs.
        
        #self.name = name
        self.ROI_list = ROI_list
        self.set_list = set_list + ['sum']
        #self.hist = ['h']
        _, self.binEdges = np.histogram([0], bins = Nbins, range = (binStart, binEnd))
        self.headr = list(itertools.product(self.ROI_list, self.set_list))
        #print (self.ROI_list, self.set_list, self.extracted, self.headr)
        #print (type(self.ROI_list), type(self.set_list), type(self.extracted), type(self.headr))
        self.cols = pd.MultiIndex.from_tuples(self.headr)
        
        self.df = pd.DataFrame([], self.binEdges, self.cols)
        print (self.df.head())
        
    def ROI_sum (self):
        
        for _ROI in list(self.df.columns.levels[0]):
            self.addHist (_ROI, "Sum", self.df[_ROI].sum(axis=1))
        
    def addHist (self, _ROI, _set, _h):
        # the histogram are arrays of values that belong to a ROI and a set (condition) or a sum for the ROI.
            
        # overwrite or add column if new
        self.df[_ROI, _set] = pd.Series(_h)
        

    def getHist (self, _ROI, _set):
        
        _hx = pd.Series(self.df.index)
        _hy = self.df[_ROI, _set]
        
        return _hx, _hy

