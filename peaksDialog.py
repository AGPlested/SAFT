import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pandas as pd


class getPeaksDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(getPeaksDialog, self).__init__(*args, **kwargs)
        
        self.peaksScraped = False
        self.small = 3
        
        self.makeDialog()
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Get peaks")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.resize(400,500)
        vbox = QVBoxLayout()
        vbox.addWidget(w)
        self.setLayout(vbox)
        
        self.N_ROI_label = QLabel('Scraping ROIs for peaks')
        layout.addWidget(self.N_ROI_label, 0, 0, 1, 2)
        
        SNR_selecter_label = QLabel('Skip ROIs with SNR less than')
        layout.addWidget(SNR_selecter_label, 1, 0, 1, 2)
        
        #SNR selecter and update N_ROI_label
        self.sbsSB = pg.SpinBox(value=3, step=.2, bounds=[1, 10], delay=0)
        self.sbsSB.setFixedSize(60, 25)
        layout.addWidget(self.sbsSB, 1, 2, 1, 1)
        self.sbsSB.valueChanged.connect(self.maskLowSNR)
        
        #will be altered as soon as data loads
        self.skipRB = QRadioButton('Skip ROIs')
        self.skipRB.setChecked(True)
        layout.addWidget(self.skipRB,2,0,1,3)
        
        psr_label = QLabel('Search range around peak (data points)')
        layout.addWidget(psr_label,  3, 0, 1, 2)
        
        self.psrSB = pg.SpinBox(value=3, step=2, bounds=[1, 7], delay=0, int=True)
        self.psrSB.setFixedSize(60, 25)
        layout.addWidget(self.psrSB, row=3, col=2)
        
        _doScrapeBtn = QPushButton('Extract responses')
        _doScrapeBtn.clicked.connect(self.scrapePeaks)
        layout.addWidget(_doScrapeBtn, 5, 0, 1, 2)
        
        _cancelBtn = QPushButton('Cancel')
        _cancelBtn.clicked.connect(self.reject)
        
        layout.addWidget(_cancelBtn, row=5, col=2)
        self.setLayout(layout)
         
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
        
        self.tracedata = data
        tdk = self.tracedata.keys()
        tdk_display = ", ".join(str(k) for k in tdk)
        N_ROI = [len (self.tracedata[d].columns) for d in tdk]
        
        self.N_ROI_label.setText("Scraping peaks from {} ROIs \n over the sets named {}".format(N_ROI, tdk_display))
        
        _printable = "{}\n{}\n".format(tdk_display, [self.tracedata[d].head() for d in tdk])
        print ("Added data of type {}:\n{}\n".format(type(self.tracedata), _printable))
        self.maskLowSNR()
        
    def scrapePeaks(self):
        """Some peak-finding function with output filtering based on SNR"""
        
        self.prepGuiParameters()
        self.pkextracted_by_set = {}
        
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
        
        if self.ignore:
            #blacklist peaks from traces with low SNR
            self.maskLowSNR()
        
            _cut = self.badSNRcut
            _blacklisted_by_set = {}
            
            #split peak data into sets from high and low SNR
            for s in self.pkextracted_by_set.keys():
                wls = self.whitelists[s]
                bls = self.blacklists[s]
                pk = self.pkextracted_by_set[s]
                whitelisted = pk[wls.sort_values(ascending=False).index]
                blacklisted = pk[bls.sort_values(ascending=False).index]
                self.pkextracted_by_set[s] = whitelisted
                _blacklisted_by_set[s + "_SNR<" + str(_cut)] = blacklisted
         
            #put the blacklisted items back as new key value pairs (marked keys will be used for Excel sheets)
            for s,bl in _blacklisted_by_set.items():
                self.pkextracted_by_set[s] = bl
        
        #close the dialog
        self.accept()
    

    def maskLowSNR(self):
        """split the peak output according to SNR of parent traces"""
        """moving the spinbox for SNR Cutoff also comes here"""
        print ('maskLowSNR')
        
        self.badSNRcut = self.sbsSB.value()

        # use selective region (LR?)
        # or just the whole trace?
        
        #store whitelists and blacklists as values with the set as key.
        self.whitelists = {}
        self.blacklists = {}
        wl_count = 0
        bl_count = 0
        
        for _set in self.tracedata:
            
            _df = self.tracedata[_set]
            
            # find SNR from column-wise Max / SD
            snr = _df.max() / _df.std()
            
            #histogram with SNRcut drawn?
            
            self.whitelists[_set] = snr.where(snr >= self.badSNRcut).dropna()
            self.blacklists[_set] = snr.where(snr < self.badSNRcut).dropna()
            
            wl_count += len(self.whitelists[_set])
            bl_count += len(self.blacklists[_set])
        
            #output to window?
            print ("Whitelist "+_set, self.whitelists[_set])
            print ("Blacklist "+_set, self.blacklists[_set])
      
        skipLabelText = "Skipping {0} traces out of {1} for low SNR.".format(bl_count, wl_count+bl_count)
        self.skipRB.setText(skipLabelText)
        
"""
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("FSTPA")
        self.central_widget = QWidget()
        self.central_layout = QGridLayout()
        self.central_widget.setLayout(self.central_layout)
        self.setCentralWidget(self.central_widget)
        self.create_control()
        self.data = np.array([1,343,33,16,2,3,4,5,6])
        self.extPa = {"Max" : 9}
        
    def create_control(self):
        
        dataBtn = QPushButton('Get peak data')
        dataBtn.clicked.connect(self.getPeakData)
        self.central_layout.addWidget(dataBtn)
        
        
    def getPeakData(self):
        ""Wrapping function to get peak data from the dialog""
        
        # if the QDialog object is instantiated in __init__, it persists in state....
        # do it here to get a fresh one each time.
        self.gpd = getPeaksDialog()
        
        # pass the data into the get peaks dialog object
        self.gpd.addData(self.data)
        
        # pass in external parameters for the peak extraction
        self.gpd.setExternalParameters(self.extPa)
        
        # returns 1 (works like True) when accept() or 0 (we take for False) otherwise. 
        accepted = self.gpd.exec_()
        
        # data from dialog is stored in its attributes
        
        if accepted:
            if self.gpd.peakidx != -1:
                print ("Peak {} scraped at index {} and self.gpd.peaksScraped is {}".format(self.gpd.peakWanted, self.gpd.peakidx, self.gpd.peaksScraped))
            else:
                print ("Peak {} not found. self.gpd.peaksScraped is {}".format(self.gpd.peakWanted, self.gpd.peaksScraped))
        else:
            print ("Returned but not happily: self.gpd.peaksScraped is {}".format(self.gpd.peaksScraped))
  """
        
        
if __name__ == '__main__':
    
    app = QApplication([])
    main_window = getPeaksDialog()
    main_window.show()
    sys.exit(app.exec_())
