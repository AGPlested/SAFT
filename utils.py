import os.path
import pyqtgraph as pg
import string
import random
import pandas as pd

def decomposeRDF(rdf):
    """decompose MultiIndex dataframe R-C-(t-pk-pairs)
    into dictionary with C as keys and dataframes as values
    Each dataframe with t as index and pk as columns named by R"""
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
    _stem = _tail.rsplit(".", 1)          #from "file.txt" -> "file", "text"
    
    return _stem[0]

def addFileSuffix(_name, _suffix):

    _split = os.path.split(_name)
    _path = _split[0]
    _tail = _split[1]
    _stem = _tail.rsplit(".", 1)     #from "file.txt" -> "file", "text"
    
    return _path + _stem[0] + _suffix + "." + _stem[1]

def getRandomString(length):
    ###https://pynative.com/python-generate-random-string/
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)

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

def removeAllScatter(p1):
    """p1 should be a plot item"""
    
    PDIs = [d for d in p1.items if isinstance(d, pg.PlotDataItem)]
    for pdi in PDIs:
        
        x, _ = pdi.scatter.getData()            #scatter data objects have some data points in them
        if len(x) > 0:
            print ("Removing: {}".format(pdi))
            p1.removeItem(pdi)                  #need to use remove item to get rid of it.
            
    _rem = p1.listDataItems()

    print("Data items remaining in {0}: {1}".format(p1, len(_rem)))
