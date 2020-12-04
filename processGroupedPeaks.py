import sys
import os.path
import itertools
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout, QFileDialog
import numpy as np
import pyqtgraph as pg
import pandas as pd

def sanitizeList(l):
    return [x.strip().replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '') for x in l]

class groupPeakDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(groupPeakDialog, self).__init__(*args, **kwargs)
        
        self.peaksN = 50    # it's a guess
        self.dataLoaded = False
        self.makeDialog()
        
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Grouped peaks analysis")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.resize(400,400)
        vbox = QVBoxLayout()
        vbox.addWidget(w)
        self.setLayout(vbox)
        
        self.NRepeatsLabel = QLabel('Analysing grouped peaks')
        layout.addWidget(self.NRepeatsLabel, 1, 0, 1, 3)
        
        self.N_ROI_label = QLabel('Grouping peaks from ROIs \n over sets')
        layout.addWidget(self.N_ROI_label, 2, 0, 1, 2)
        
        groupNSB_selecter_label = QLabel('Number of responses in each group')
        layout.addWidget(groupNSB_selecter_label, 0, 0, 1, 2)
        
        # N peaks in each repeating group
        # (that is, N=5 means the peaks are in groups of 5, not 5 groups)
        self.groupNSB = pg.SpinBox(value=5, step = 1, bounds=[1, 10], delay=0)
        self.groupNSB.setFixedSize(60, 25)
        layout.addWidget(self.groupNSB, 0, 2, 1, 1)
        self.groupNSB.valueChanged.connect(self.updateGrouping)
       
        _doScrapeBtn = QPushButton('Extract grouped data')
        _doScrapeBtn.clicked.connect(self.groupPeaks)
        layout.addWidget(_doScrapeBtn, 5, 0, 1, 2)
        
        self.saveGraphsBtn = QPushButton('Save graphs of grouped peak responses')
        
        # at first, there is nothing to save
        self.saveGraphsBtn.setEnabled(False)
        self.saveGraphsBtn.clicked.connect(self.saveGraphs)
        layout.addWidget(self.saveGraphsBtn, 7, 0, 1, 2)
        
        self.saveGroupsBtn = QPushButton('Save grouped peak data')
        
        # at first, there is nothing to save
        self.saveGroupsBtn.setEnabled(False)
        self.saveGroupsBtn.clicked.connect(self.saveGroupPeaks)
        layout.addWidget(self.saveGroupsBtn, 6, 0, 1, 2)
        
        _doneBtn = QPushButton('Done')
        _doneBtn.clicked.connect(self.gPDone)
        
        layout.addWidget(_doneBtn, row=6, col=2)
        self.setLayout(layout)
        
        self.updateGrouping()
    
    def gPDone (self, *args):
        try:
            print ("Done.")
            self.accept()   # works if the dialog was called from elsewhere
        except:
            print ("Bye.")
            self.hide()     # works if the dialog was called standalone
    
    def saveGroupPeaks (self, *args, verbose=True):
        
        if verbose: print ("Open dialog to save grouped peak data.")
            #print (self.hDF.df.head(5))
        
        # below needs to be updated for openpyxl
        
        # format for header cells.
        # self.hform = {
        #'text_wrap': True,
        #'valign': 'top',
        #'fg_color': '#A504AC',
        #'border': 1}
        
        self.filename = QFileDialog.getSaveFileName(self,
        "Save Group Analysis", os.path.expanduser("~"))[0]
        
        # from XlsxWriter examples, John McNamara
        # updated for openpyxl
        
        if self.filename:
            with pd.ExcelWriter(self.filename) as writer:
                #save peaks into sheet
                for _set, _df in self.groupsextracted_by_set.items():
                
                    _df.to_excel(writer, sheet_name=_set, startrow=1, header=False)
                    #_workbook  = writer.book
                    _worksheet = writer.sheets[_set]
                    #header_format = _workbook.add_format(self.hform)
                    for col_num, v in enumerate(_df.columns.values):
                        #print (col_num, value, _set)
                        # redone for openpyxl syntax
                        _worksheet.cell(1, col_num + 2).value = str(v[0]) + " " + _set
                        _worksheet.cell(2, col_num + 2).value = str(v[1])
           
            if verbose: print ("Saved {}".format(self.filename))
        
        else:
            if verbose: print ("No filename given, nothing saved.")
           
    def saveGraphs(self):
        """Multipage PDF output of groupData analysis"""
        self.gfilename = QFileDialog.getSaveFileName(self,
        "Save Group Analysis figures", os.path.expanduser("~"))[0]
      
        if self.gfilename:
            # from https://stackoverflow.com/a/31133424 Victor Juliet
            import datetime
            
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            import matplotlib
            from matplotlib.font_manager import FontProperties
            
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            
            font = FontProperties()
            #font.set_family('serif')
            font.set_name('Helvetica')

            # Create the PdfPages object to which we will save the pages:
            # The with statement makes sure that the PdfPages object is closed properly at
            # the end of the block, even if an Exception occurs.
            
            #df_list = [ v for k,v in self.groupsextracted_by_set.items()]
            fixedKeys = sanitizeList(self.groupsextracted_by_set.keys())
            merged = pd.concat(self.groupsextracted_by_set.values(), axis=1, keys=fixedKeys)
            
            #merged.columns.levels[1] = merged.columns.levels[1]
            #print (merged.shape)
            #print (merged)
            #print (merged.columns.values)
            #print (merged.columns.levels[2])
            #print (merged.columns.levels[1])
            #print (merged.columns.levels[0])
            All = slice(None)
            
            with PdfPages(self.gfilename+'.pdf') as pdf:
            
                for _ROI in list(merged.columns.levels[1]):
                # loop over ROIs
                # for each, if a group column exists then plot and use SD for shading
                # title by ROI, 6 per page?
                    plt.figure(figsize=(4, 3))
                    plt.axes(ylim=(0, 1))
                    plt.xticks(fontname = "Helvetica")
                    plt.yticks(fontname = "Helvetica")
                    plt.title("ROI " + _ROI, fontproperties=font)
                    
                    for _set in list(merged.columns.levels[0]):
                        
                        #some indexes are not there
                        if (_set, _ROI, 'mean') in list(merged.columns.values):
                            SD = merged.loc[All, (_set, _ROI, 'SD')]
                            m = merged.loc[All, (_set, _ROI, 'mean')]
                        
                            yn = pd.to_numeric(m-SD)
                            yp = pd.to_numeric(m+SD)
                        
                            plt.plot(range(self.step), m, 'o-')
                            #plt.plot(range(self.step), m, 'k')
                            plt.fill_between([float(x) for x in range(self.step)], yn, yp, alpha=0.5, facecolor='#FF9848') #edgecolor='#CC4F1B',
                        
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = 'PDF of graphs from SAFT'
                #d['Author']
                d['Subject'] = 'group peak analysis'
                #d['CreationDate']
                d['ModDate'] = datetime.datetime.today()
            print ("PDF output complete.")
    
    def setExternalParameters(self, extPa):
        """extPa is a dictionary of external parameters that can be passed"""
        if 'Max' in extPa:
            self.selecter.setMaximum(extPa['Max'])
            print ("Changing spinbox max to : {}".format(extPa['Max']))
        if 'Min' in extPa:
            self.selecter.setMinimum(extPa['Min'])
            print ("Changing spinbox min to : {}".format(extPa['Min']))
        if 'tPeaks' in extPa:
            self.tPeaks = extPa['tPeaks']
        
    def prepGuiParameters(self, verbose=True):
        """Take parameters specified by GUI"""
        if verbose: print ("Take parameters from GUI and prepare it?")
        #True if box is checked, otherwise False
        #self.ignore =  self.skipRB.isChecked()
        
        #self.psr = self.psrSB.value() // 2          #floor division to get ears
    
    #not using this currently
    def addDataset(self, d):
        """Bring in external dataset for analysis"""
        self.peakData = d.resultsDF     #resultsDF object
        self.name = d.name
        print (self.peakData)
        
        # the following was designed for a dictionary, maybe fails with resultsDF object
        # remove any duplicate peaks
        for k, _v in self.peakData.items():
            _wasShape = _v.shape
            self.peakData[k] = _v.loc[~_v.index.duplicated(keep='first')] #StackOverflow 13035764
            _isShape = _v.shape
            if _isShape != _wasShape:
                print ("Removed duplicates, df.shape() was {}, now {}".format(_wasShape, _isShape))
            
        pdk = self.peakData.keys()
        pdk_display = ", ".join(str(k) for k in pdk)
        N_ROI = [len (self.peakData[d].columns) for d in pdk]
        
        _printable = "{}\n{}\n".format(pdk_display, [self.peakData[d].head() for d in pdk])
        print ("Added data {} of type {}:\n{}\n".format(self.name, type(self.peakData), _printable))
        
        self.dataLoaded = True
        self.N_ROI_label.setText("Grouping peaks from {} ROIs \n over the sets named {}".format(N_ROI, pdk_display))
        self.updateGrouping()
        
    def addData(self, data, *args, name=None, verbose=True):
        """Bring in external data for analysis"""
        
        print ("Add (import) data.")
        
        # data is expected to be a dictionary of Pandas DataFrames
        self.peakData = data
        if name:
            self.name = name
        else:
            self.name = "unnamed"
 
        #remove any duplicate peaks
        for k, _v in self.peakData.items():
            _wasShape = _v.shape
            self.peakData[k] = _v.loc[~_v.index.duplicated(keep='first')] # StackOverflow 13035764
            _isShape = _v.shape
            if _isShape != _wasShape:
                if verbose: print ("Removed duplicates, df.shape() was {}, now {}".format(_wasShape, _isShape))
            
        pdk = self.peakData.keys()
        pdk_display = ", ".join(str(k) for k in pdk)
        N_ROI = [len (self.peakData[d].columns) for d in pdk]
        
        _printable = ""
        for k, v in self.peakData.items():
            _printable += ("{}\n{}\n.....\n".format(k, v.head()))
        
        if verbose: print ("Added self.peakData of type: {}\n{}".format(type(self.peakData), _printable))
        
        self.dataLoaded = True
        self.N_ROI_label.setText("Grouping peaks from {} dataset: {} ROIs \n over the conditions {}".format(self.name, N_ROI, pdk_display))
        self.updateGrouping()
        
        
        
    def groupPeaks(self, *args, verbose=True):
        print ("Analyse group peaks.")
        
        self.prepGuiParameters()
        self.groupsextracted_by_set = {}
        #_step  = int(self.groupNSB.value())
        All = slice(None)
       
        
        for _set in self.peakData.keys():
            # prep means and sd frames
            _c = self.peakData[_set].columns
            if verbose: print("_c, self.step \n{}\t{}".format(_c, self.step))
            
            
            _stat = ['mean','SD']
            _headr = list(itertools.product(_c, _stat))
            # make dataframe
            cols = pd.MultiIndex.from_tuples(_headr)
            _s = pd.DataFrame([], range(self.step), cols)
            if verbose: print ("_s: {}".format(_s))
            
            for p in range(self.step):
                # get pth row group
                _subset = self.peakData[_set].iloc[p::self.step]
                if verbose: print ("{0}. subset\n {1}\n".format(p + 1, _subset))
                
                _ssmean = _subset.describe().loc['mean']
                _ssSD = _subset.describe().loc['std']
                # each set of paired mean, sd results is assigned to two columns
                
                pth_row = _s.index.isin(_s.index[p:p+1])
                if verbose: print ("pth row: {0}\nempty\n {1}\n describe.transposed\n{2}\n".format(pth_row, _s.loc[pth_row, (All, 'mean')] , _ssmean.transpose()))
                
                _ROI_cols = [str(x) for x in _ssmean.index.values]
                
                _s.loc[pth_row, (_ROI_cols, 'mean')] = _ssmean.values       
                _s.loc[pth_row, (_ROI_cols, 'SD')] = _ssSD.values

            self.groupsextracted_by_set[_set] = _s
        
        # we can save now that we have data
        self.saveGroupsBtn.setEnabled(True)
        self.saveGraphsBtn.setEnabled(True)
        
        print ("\n\nResults of group analysis\n-------------\n")
        for k,v in self.groupsextracted_by_set.items():
            print ("{}\n{}\n".format(k,v))
        

    def updateGrouping(self, *args, verbose=False):
        """recalculate groups"""
        
        print ('Update peak grouping.')
        
        if self.dataLoaded:
            try:
                _firstdf = list(self.peakData.values())[0]
                if verbose: print ("first DF,\n", _firstdf)
                self.peaksN = _firstdf.shape[0]     # get the number of rows in the first DataFrame
            except:
                if verbose: print ("couldn't find a dataframe in self.peakData")
                return False
                
        self.NRepeats  = int(self.peaksN / self.groupNSB.value())
        self.step = int(self.groupNSB.value())
        #update dialog
        NrepeatsLabelText = "{0} peaks in each trace and so {1} repeats.".format(self.peaksN, self.NRepeats)
        self.NRepeatsLabel.setText(NrepeatsLabelText)
        

        
        
if __name__ == '__main__':
    
    app = QApplication([])
    main_window = groupPeakDialog()
    
    # trial data
    a = np.arange(150).reshape((50, 3))
    b = np.arange(150).reshape((50, 3))
    np.random.shuffle(b)
    c = ['a', 'b', 'c']
    
    d = {
        "Set One" : pd.DataFrame(a**2, columns=c),
        "Set Two" : pd.DataFrame(b, columns=c)
        }
    main_window.addData(d)
        
    main_window.show()
    sys.exit(app.exec_())
