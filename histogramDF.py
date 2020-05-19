import itertools
import pandas as pd
import numpy as np

class HistogramsR():
    """ a data frame for the common histograms result """

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
        
    def addHist (self, _ROI, _set, _h):
        # the histogram are arrays of values that belong to a ROI and a set (condition) or a sum for the ROI.
        #_h = np.append(_h, [np.nan])       #to equalize lengths
        
        print (_ROI, _set, _h)
        print (len(self.df.index), len(_h))
        # overwrite or add column if new
        self.df[_ROI, _set] = pd.Series(_h)
        print (self.df[_ROI][_set])
        

    def getHist (self, _ROI, _set):
        
        _hx = self.binEdges
        _hy = self.df[_ROI, _set]
        
        return _hy, _hx #to match np.histogram

