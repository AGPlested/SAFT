import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pandas as pd


class groupDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(groupDialog, self).__init__(*args, **kwargs)
        
        #group = 5
        self.peaksN = 50
        
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
        
        groupNSB_selecter_label = QLabel('Number of grouped responses')
        layout.addWidget(groupNSB_selecter_label, 0, 0, 1, 2)
        
        # group N selecter
        self.groupNSB = pg.SpinBox(value=5, step = 1, bounds=[1, 10], delay=0)
        self.groupNSB.setFixedSize(60, 25)
        layout.addWidget(self.groupNSB, 0, 2, 1, 1)
        self.groupNSB.valueChanged.connect(self.updateGroup)
        
        #will be altered as soon as data loads
        self.skipRB = QRadioButton('Skip ROIs')
        self.skipRB.setChecked(True)
        layout.addWidget(self.skipRB,2,0,1,3)
        
        psr_label = QLabel('Search range around peak (data points)')
        layout.addWidget(psr_label,  3, 0, 1, 2)
        
        self.psrSB = pg.SpinBox(value=3, step=2, bounds=[1, 7], delay=0, int=True)
        self.psrSB.setFixedSize(60, 25)
        layout.addWidget(self.psrSB, row=3, col=2)
        
        _doScrapeBtn = QPushButton('Extract grouped data')
        _doScrapeBtn.clicked.connect(self.groupPeaks)
        layout.addWidget(_doScrapeBtn, 5, 0, 1, 2)
        
        _saveGroupsBtn = QPushButton('Save grouped peak data')
        _saveGroupsBtn.clicked.connect(self.saveGroupPeaks)
        layout.addWidget(_saveGroupsBtn, 6, 0, 1, 2)
        
        _cancelBtn = QPushButton('Done')
        _cancelBtn.clicked.connect(self.accept)
        
        layout.addWidget(_cancelBtn, row=6, col=2)
        self.setLayout(layout)
        
        self.updateGroup()
    
    def saveGroupPeaks (self):
        print ("save group peak data")
       
    
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
        
        #True if box is checked, otherwise False
        self.ignore =  self.skipRB.isChecked()
        
        self.psr = self.psrSB.value() // 2          #floor division to get ears
    
    
    def addData(self, data):
        """Bring in external data for analysis"""
        
        self.peakData = data
        pdk = self.peakData.keys()
        pdk_display = ", ".join(str(k) for k in pdk)
        N_ROI = [len (self.peakdata[d].columns) for d in pdk]
        
        self.N_ROI_label.setText("Grouping peaks from {} ROIs \n over the sets named {}".format(N_ROI, pdk_display))
        
        _printable = "{}\n{}\n".format(pdk_display, [self.peakdata[d].head() for d in pdk])
        print ("Added data of type {}:\n{}\n".format(type(self.peakdata), _printable))
        
        
    def groupPeaks(self):
        print ("Analayse group peaks.")
        
        self.prepGuiParameters()
        self.groupsextracted_by_set = {}
        """
        for _set in self.tracedata.keys():
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
        
    
        # yes, output may be modified below
        self.peaksScraped = True
        self.blacklisted_by_set = {}
        
        if self.ignore:
            
            # freshly blacklist peaks from traces with low SNR
            self.maskLowSNR()
        
            # the cut-off value
            _cut = self.badSNRcut
            
            #split peak data into sets from high and low SNR
            for s in self.pkextracted_by_set.keys():
                wls = self.whitelists[s]
                bls = self.blacklists[s]
                pk = self.pkextracted_by_set[s]
                whitelisted = pk[wls.sort_values(ascending=False).index]
                blacklisted = pk[bls.sort_values(ascending=False).index]
                self.pkextracted_by_set[s] = whitelisted
                self.blacklisted_by_set[s + "_SNR<" + str(_cut)] = blacklisted
    
        #close the dialog
        self.accept()"""
    

    def updateGroup(self):
        """recalculate groups"""
        
        print ('update group')
        
        self.NRepeats  = self.peaksN / self.groupNSB.value()
        
        #update dialog
        NrepeatsLabelText = "{0} peaks in each trace and so {1: .1f} repeats.".format(self.peaksN, self.NRepeats)
        self.NRepeatsLabel.setText(NrepeatsLabelText)
        

        
        
if __name__ == '__main__':
    
    app = QApplication([])
    main_window = groupDialog()
    main_window.show()
    sys.exit(app.exec_())
