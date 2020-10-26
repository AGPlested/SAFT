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

class groupDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(groupDialog, self).__init__(*args, **kwargs)
        
        self.peaksN = 50    # it's a guess
        self.dataLoaded = False
        self.makeDialog()
        
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Group analysis")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.resize(400,500)
        vbox = QVBoxLayout()
        vbox.addWidget(w)
        self.setLayout(vbox)
        
        self.NRepeatsLabel = QLabel('Analysing grouped peaks')
        layout.addWidget(self.NRepeatsLabel, 1, 0, 1, 3)
        
        self.N_ROI_label = QLabel('Grouping peaks from ROIs \n over sets')
        layout.addWidget(self.N_ROI_label, 2, 0, 1, 2)
        
        groupNSB_selecter_label = QLabel('Number of responses in group')
        layout.addWidget(groupNSB_selecter_label, 0, 0, 1, 2)
        
        # group N selecter
        self.groupNSB = pg.SpinBox(value=5, step = 1, bounds=[1, 10], delay=0)
        self.groupNSB.setFixedSize(60, 25)
        layout.addWidget(self.groupNSB, 0, 2, 1, 1)
        self.groupNSB.valueChanged.connect(self.updateGrouping)
       
        _doScrapeBtn = QPushButton('Extract grouped data')
        _doScrapeBtn.clicked.connect(self.groupPeaks)
        layout.addWidget(_doScrapeBtn, 5, 0, 1, 2)
        
        self.saveGraphsBtn = QPushButton('Save graphs of grouped peaks')
        
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
        _doneBtn.clicked.connect(self.accept)
        
        layout.addWidget(_doneBtn, row=6, col=2)
        self.setLayout(layout)
        
        self.updateGrouping()
    
    def saveGroupPeaks (self):
        print ("save group peak data")
        #print (self.hDF.df.head(5))
        
        # format for header cells.
        self.hform = {
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#A504AC',
        'border': 1}
        
        self.filename = QFileDialog.getSaveFileName(self,
        "Save Group Analysis", os.path.expanduser("~"))[0]
        
        #from XlsxWriter examples, John McNamara
        if self.filename:
            with pd.ExcelWriter(self.filename) as writer:
                #save peaks into sheet
                for _set, _df in self.groupsextracted_by_set.items():
                
                    _df.to_excel(writer, sheet_name=_set, startrow=1, header=False)
                    _workbook  = writer.book
                    _worksheet = writer.sheets[_set]
                    header_format = _workbook.add_format(self.hform)
                    for col_num, value in enumerate(_df.columns.values):
                        #print (col_num, value, _set)
                        _worksheet.write(0, col_num + 1, str(value) + " " + _set, header_format)
                    
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
        
    def prepGuiParameters(self):
        """Take parameters specified by GUI"""
        print ("prep GUI parameters?")
        #True if box is checked, otherwise False
        #self.ignore =  self.skipRB.isChecked()
        
        #self.psr = self.psrSB.value() // 2          #floor division to get ears
    
    
    def addDataset(self, d):
        """Bring in external dataset for analysis"""
        self.peakData = d.resultsDF     #resultsDF object
        self.name = d.name
        print (self.peakData)
        
        #following was designed for dictionary, maybe fails with resultsDF object
        #remove any duplicate peaks
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
        
    def addData(self, data):
        """Bring in external data for analysis"""
        #data is a dictionary of Pandas DataFrames
        self.peakData = data
        print (self.peakData)
        
        #remove any duplicate peaks
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
        print ("Added data of type {}:\n{}\n".format(type(self.peakData), _printable))
        
        self.dataLoaded = True
        self.N_ROI_label.setText("Grouping peaks from {} ROIs \n over the sets named {}".format(N_ROI, pdk_display))
        self.updateGrouping()
        
    def groupPeaks(self):
        print ("Analayse group peaks.")
        
        self.prepGuiParameters()
        self.groupsextracted_by_set = {}
        #_step  = int(self.groupNSB.value())
        All = slice(None)
       
        
        for _set in self.peakData.keys():
            #prep means and sd frames
            _c = self.peakData[_set].columns
            #print (c, _step)
            
            
            _stat = ['mean','SD']
            _headr = list(itertools.product(_c, _stat))
            # make dataframe
            cols = pd.MultiIndex.from_tuples(_headr)
            _s = pd.DataFrame([], range(self.step), cols)
            print (_s)
            
            for p in range(self.step):
                # get pth row group
                _subset = self.peakData[_set].iloc[p::self.step]
                print ("{0}. subset\n {1}".format(p, _subset))
                # each set of paired mean, sd results is assigned to two columns
                pth_row = _s.index.isin(_s.index[p:p+1])
                #print (pth_row, _s.loc[pth_row, (All, 'mean')] , _subset.describe().loc['mean'])
                _s.loc[pth_row, (All, 'mean')] = _subset.describe().loc['mean'].values
                _s.loc[pth_row, (All, 'SD')] = _subset.describe().loc['std'].values
    
                #_means.iloc[p] = _subset.describe().loc['mean']
                #_SDs.iloc[p] = _subset.describe().loc['std']
            
            #print(_s)
            self.groupsextracted_by_set[_set] = _s
            #self.groupsextracted_by_set[_set + "_sd"] = _SDs
        
        # we can save now that we have data
        self.saveGroupsBtn.setEnabled(True)
        self.saveGraphsBtn.setEnabled(True)
        print (self.groupsextracted_by_set)
        """
        for _set in self.peakdata.keys():
            maxVal = len(self.tPeaks)
        
        
            ROI_df = self.tracedata[_set]
            #print (ROI_df)
        
            peaksList = []
            progMsg = "Get {0} peaks, {1} set..".format(maxVal, _set)
            with pg.ProgressDialog(progMsg, 0, maxVal) as dlg:
                for t in self.tPeaks:
                    dlg += 1
                    idx =  np.searchsorted(ROI_df.index, t)
                    # avoid falling off start or end of columns
                    e = max (idx-self.psr, 0)
                    # zero biased so add one.
                    l = min (idx+self.psr+1, len(ROI_df.index))
                    print (t, e, idx, l, ROI_df.iloc[e:l])
                    
                    p = ROI_df.iloc[e:l].max().to_frame().transpose()
                    peaksList.append(p)
                
                #stick rows together (all have same zero index...)
                peaksdf = pd.concat(peaksList)
                
                # Overwrite index with the original peak positions
                # (somewhat inexact because of the 'range')
                peaksdf.index = self.tPeaks
                self.pkextracted_by_set[_set] = peaksdf
        
    
        
        #close the dialog
        self.accept()"""
    

    def updateGrouping(self):
        """recalculate groups"""
        
        print ('update grouping')
        
        if self.dataLoaded:
            try:
                _firstdf = list(self.peakData.values())[0]
                print (_firstdf)
                self.peaksN = _firstdf.shape[0]     # get the number of rows in the first DataFrame
            except:
                print ("couldn't find a dataframe in self.peakData")
                
        self.NRepeats  = int(self.peaksN / self.groupNSB.value())
        self.step = int(self.groupNSB.value())
        #update dialog
        NrepeatsLabelText = "{0} peaks in each trace and so {1} repeats.".format(self.peaksN, self.NRepeats)
        self.NRepeatsLabel.setText(NrepeatsLabelText)
        

        
        
if __name__ == '__main__':
    
    app = QApplication([])
    main_window = groupDialog()
    
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
