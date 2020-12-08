### SAFT Utility Functions
### Andrew Plested 2020-11-14

import sys
import os.path
import string
import random

import pandas as pd
import numpy as np

import pyqtgraph as pg

from PySide2 import QtGui

class txOutput():
    """Console frame"""
    def __init__(self, initialText, *args, **kwargs):
        
        self.text = initialText
        self.frame = QtGui.QTextEdit()
        font = QtGui.QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.frame.setCurrentFont(font)
        self.appendOutText(initialText)
        self.frame.setReadOnly(True)
        self.size()

    def appendOutText(self, newOP=None, color="Black"):
        self.frame.setTextColor(color)
        if newOP != None:
            self.frame.append(str(newOP))

    def size(self, _w=200, _h=200):
        self.frame.resize(_w, _h)
        self.frame.setMinimumSize(_w, _h)
        self.frame.setMaximumSize(_w, _h)
    
    def reset(self, initialText):
        self.frame.clear()
        self.appendOutText(initialText)


def extendMaskArray(series, r):
    """series should be a list of indices, r is the width"""
    new = []
    for i in series:
        new.extend(np.arange(i-r, i+r))
    return  pd.Series(new).drop_duplicates()
    
def maskPeaks(df, peaks, width):
    """df is the dataframe to mask rows
    peaks are the row indices
    width is the extent to mask around each row in peaks"""
    e = extendMaskArray(peaks, width)
    # make sure no values are outside the row count (although pandas would not care
    f = pd.Series(e.values[(e>=0) & (e<len(df.index))])

    return df[~df.index.isin (f)]

def decomposeRDF(rdf):
    """
    Decompose a pandas MultiIndex dataframe ('rdf') with levels R, C, (time-peak pairs)
    into dictionary ('decomposed') with C as keys and dataframes as values
    Each dataframe has time as index and peaks as columns named by R
    """
    decomposed = {}
    
    for co in rdf.columns.get_level_values(1).unique():
        # take the sub-dataframe for each condition
        inter = rdf.loc(axis=1)[:,co,:]
        # remove condition index
        inter.columns = inter.columns.droplevel(1)
        # set first 't' column as index
        inter.set_index(inter.columns[0])
        # remove 't' columns
        inter = inter.drop(columns='t', level = 1)
        # remove 'p' level
        inter.columns = inter.columns.droplevel(1)
        decomposed [co] = inter

    return decomposed

def getFileStem(_name):

    _split = os.path.split(_name)
    _tail = _split[1]
    _stem = _tail.rsplit(".", 1)            #"file.txt" -> "file", "text"
    
    return _stem[0]

def addFileSuffix(_name, _suffix):

    _split = os.path.split(_name)
    _path = _split[0]
    _tail = _split[1]
    _stem = _tail.rsplit(".", 1)             #"file.txt" -> "file", "text"
    
    return _path + _stem[0] + _suffix + "." + _stem[1]

def getRandomString(length):
    ###https://pynative.com/python-generate-random-string/
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
    

def linePrint(results, pre=2, pitch=7):
    """
    changes a list of results into consistently-spaced, readable string
    pre     : the precision of any floating point value
    pitch   : the field spacing
    """
    
    readable = ""
    
    for item in results:
        if isinstance(item, str):
            readable += "{:^{pi}}".format(item, pi=pitch)
        elif isinstance(item, float):
            readable += "{:{pi}.{prec}f}".format(item, prec=pre, pi=pitch)
        else:
            try:
                readable += "{:^{pi}}".format(str(item), pi=pitch)
            except:
                readable += "<error>"
                print("Conversion error:", sys.exc_info()[0])
                raise
                
    return readable

def findCurve(items):
    # assume there is one PG PlotDataItem with curve data and return it
    # the others should be empty
    PDIs = [d for d in items if isinstance(d, pg.PlotDataItem)]

    # there should be two plot data items, find the curve data
    for pdi in PDIs:
        x, _ = pdi.curve.getData()
        if len(x) > 0:
            return pdi.curve

def findScatter(items):
    # assume there is one PG PlotDataItem with scatter data and return it
    # the other scatter attributes should be empty
    PDIs = [d for d in items if isinstance(d, pg.PlotDataItem)]
    
    # there should be two plot data items, find the scatter data
    for pdi in PDIs:
        x, _ = pdi.scatter.getData()
        if len(x) > 0:
            return pdi.scatter

def removeAllScatter(p1, verbose=True):
    """p1 should be a pg plotItem"""
    
    PDIs = [d for d in p1.items if isinstance(d, pg.PlotDataItem)]
    for pdi in PDIs:
        
        x, _ = pdi.scatter.getData()            #scatter data objects have some data points in them
        if len(x) > 0:
            if verbose: print ("Removing: {}".format(pdi))
            p1.removeItem(pdi)                  #need to use remove item to get rid of it.
            
    _rem = p1.listDataItems()

    if verbose:
        print("Data items remaining in {0}: {1}".format(p1, len(_rem)))
