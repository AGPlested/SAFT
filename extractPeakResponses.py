import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QCheckBox, QLayout, QDialog, QLabel, QPushButton, QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pandas as pd
import utils


class extractPeaksDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(extractPeaksDialog, self).__init__(*args, **kwargs)
        
        self.peaksScraped = False
        self.small = 3
        self.failures_nulled = False
        self.rundownThreshold = 0.5     # to assess whether responses are running down (by factor 2 or more)
        self.makeDialog()
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Extract peaks according to reference pattern")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.resize(500,400)
        vbox = QVBoxLayout()
        vbox.addWidget(w)
        self.setLayout(vbox)
        
        self.N_ROI_label = QLabel('Extracting peaks')
        
        
        #will be altered as soon as data loads
        self.skipRB = QCheckBox('Skip ROIs with SNR less than')
        self.skipRB.setChecked(False)
        self.skipRB.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.skipRB.stateChanged.connect(self.maskLowSNR)
        
        self.skipSB = pg.SpinBox(value=3, step=.2, bounds=[1, 10], delay=0)
        self.skipSB.setFixedSize(60, 25)
        self.skipSB.valueChanged.connect(self.maskLowSNR)
        
        psr_label = QLabel('Search range around peak (data points)')
        self.psrSB = pg.SpinBox(value=3, step=2, bounds=[1, 7], delay=0, int=True)
        self.psrSB.setFixedSize(60, 25)
        
        #will be altered as soon as data loads
        self.noiseRB = QCheckBox('Treat peaks as failures when < SD x')
        self.noiseRB.setChecked(False)
        self.noiseRB.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.noiseRB.stateChanged.connect(self.setFailures)
        self.noiseRB.setDisabled(True)
        

        self.noiseSB = pg.SpinBox(value=1.5, step=.1, bounds=[.2, 10], delay=0)
        self.noiseSB.setFixedSize(60, 25)
        self.noiseSB.valueChanged.connect(self.setFailures)
        self.noiseSB.setDisabled(True)
                
        self.peaksLabel = QLabel('No peaks set to be failures.')
        
        _doScrapeBtn = QPushButton('Extract responses')
        _doScrapeBtn.clicked.connect(self.scrapePeaks)
        
        self.getRundownBtn = QPushButton('Calculate rundown')
        self.getRundownBtn.clicked.connect(self.getRundown)
        self.getRundownBtn.setDisabled(True)
        
        _cancelBtn = QPushButton('Cancel')
        _cancelBtn.clicked.connect(self.reject)
        
        self.acceptBtn = QPushButton('Accept and Return')
        self.acceptBtn.clicked.connect(self.prepareAccept)
        self.acceptBtn.setDisabled(True)
        
        layout.addWidget(self.N_ROI_label, 0, 0, 1, 2)
        
        layout.addWidget(self.skipRB, 1, 0, 1, 3)
        layout.addWidget(self.skipSB, 1, 2, 1, 1)
        
        layout.addWidget(psr_label,  2, 0, 1, 2)
        layout.addWidget(self.psrSB, row=2, col=2)
        
        layout.addWidget(self.noiseRB, 3, 0, 1, 3)
        layout.addWidget(self.noiseSB, 3, 2, 1, 1)
        layout.addWidget(self.peaksLabel, 4, 0, 1, -1)
        
        layout.addWidget(_doScrapeBtn, 5, 0)
        layout.addWidget(self.getRundownBtn, 5, 1)
        layout.addWidget(_cancelBtn, 6, 1)
        layout.addWidget(self.acceptBtn, 6, 2)
        
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
    
    
    def addDataset(self, data):
        """Bring in external dataset for analysis"""
        
        self.tracedata = data.traces  # each dataframe in this dictionary could have a different set of ROI
        self.name = data.DSname + "_expk"
        tdk = self.tracedata.keys()
        tdk_display = ", ".join(str(k) for k in tdk)
        N_ROI = np.array([len (self.tracedata[d].columns) for d in tdk])
        N_Peaks = len(self.tPeaks)
        self.total_peaks = (N_ROI * N_Peaks).sum()
        self.N_ROI_label.setText("Extracting {} peaks from {} ROIs (total {})\n over the sets named {}".format(N_Peaks, N_ROI, self.total_peaks, tdk_display))
        
        _printable = "{}\n{}\n".format(tdk_display, [self.tracedata[d].head() for d in tdk])
        print ("Added data of type {}:\n{}\n".format(type(self.tracedata), _printable))
        self.maskLowSNR()
       
    def getRundown(self, silent=False):
        rtext = ""
        self.rundownCount = 0
        for _condi, _pkdf in self.pk_extracted_by_condi.items():
            
            _Np = len(_pkdf.index)
            _NROI = len(_pkdf.columns)
            ten_percent = int(_Np / 10)
            rtext += "{} condition, 10% of peaks count is {} peaks\n".format(_condi, ten_percent)
            # look at first 5 peaks
            _firsttenpc = _pkdf.iloc[0:ten_percent].describe().loc["mean"]
            # look at last 5 peaks
            _lasttenpc = _pkdf.iloc[-1-ten_percent:-1].describe().loc["mean"]
            
            _tdf = self.tracedata[_condi]
            _max = _tdf.max()
            _SD = _tdf.std()
            
            _bestSNR = _max / _SD
            _startSNR = _firsttenpc / _SD
            _endSNR = _lasttenpc / _SD
            #print ("ff, lf : {} {}".format(_firstfive, _lastfive))
            
            _rundownRatio = _lasttenpc.div(_firsttenpc).sort_values()
            self.rundownCount += _rundownRatio[_rundownRatio < self.rundownThreshold].count()
            
            _rd_SNR = pd.concat([_rundownRatio, _bestSNR, _startSNR, _endSNR], axis=1)
            _rd_SNR.columns = ['Rundown', 'Best SNR', 'Initial SNR', 'Final SNR']
            
            rtext += "Rundown (amplitude ratio: last 10% / first 10%) and signal to noise ratio (start, end)\nfor {} ROIs (ROIs with worst rundown first):\n{}\n\n".format(_NROI, _rd_SNR.round(2).to_string())
            
        rtext += "Total number of traces with rundown worse than threshold ({}): {}\n".format(self.rundownThreshold, self.rundownCount)
        
        print (rtext)
        
        if not silent:
            ###Make a pop up window of these results
            qmb = QDialog()
            qmb.setWindowTitle('Rundown {}'.format(self.name))
            qmb.setGeometry(800,600,600,600)
            self.rundownText = QtGui.QTextEdit()
            font = QtGui.QFont()
            font.setFamily('Courier')
            font.setFixedPitch(True)
            font.setPointSize(12)
            self.rundownText.setCurrentFont(font)
            self.rundownText.setText(rtext)
            self.rundownText.setReadOnly(True)
            
            #add buttons, make it the right size
            qmb.layout = QVBoxLayout()
            qmb.layout.addWidget(self.rundownText)
            qmb.setLayout(qmb.layout)
            qmb.exec_()
            
        
    def scrapePeaks(self):
        """Some peak-finding function with output filtering based on SNR"""
        
        self.prepGuiParameters()
        self.pk_extracted_by_condi = {}
        
        for _condi in self.tracedata.keys():
            maxVal = len(self.tPeaks)
        
        
            ROI_df = self.tracedata[_condi]
            #print (ROI_df)
        
            peaksList = []
            progMsg = "Get {0} peaks, {1} set..".format(maxVal, _condi)
            with pg.ProgressDialog(progMsg, 0, maxVal) as dlg:
                dlg.setMinimumWidth(300)
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
                self.pk_extracted_by_condi[_condi] = peaksdf
        
    
        # yes, output may be modified below
        self.peaksScraped = True
        self.acceptBtn.setEnabled(True)
        self.getRundownBtn.setEnabled(True)
        self.noiseRB.setEnabled(True)
        self.noiseSB.setEnabled(True)
        self.excludedListedByCondi = {}
        
        if self.ignore:
            
            # freshly excludedList peaks from traces with low SNR
            self.maskLowSNR()
            self.splitAllowedExcluded()
            
    
    def splitAllowedExcluded(self):
        # the cut-off value
        _cut = self.excludeSNRcut
            
        #split peak data into sets from high and low SNR
        for s in self.pk_extracted_by_condi.keys():
            wls = self.allowedLists[s]
            bls = self.excludedLists[s]
            pk = self.pk_extracted_by_condi[s]
            allowedListed = pk[wls.sort_values(ascending=False).index]
            excludedListed = pk[bls.sort_values(ascending=False).index]
            
            self.pk_extracted_by_condi[s] = allowedListed
            self.excludedListedByCondi[s + "_SNR<" + str(_cut)] = excludedListed
            
    def prepareAccept(self):
        
        # substitute failures (destructively)
        if self.failures_nulled:
            self.pk_extracted_by_condi = self.pk_extracted_with_failures
        
        # divide peak results into Allowed list and Excluded list based on SNR
        self.splitAllowedExcluded()
        
        self.accept()
    
    def setFailures(self):
        self.failures_nulled = True
        # setting failures modifies the output destructively and must retain original!!!
        self.pk_extracted_with_failures = self.pk_extracted_by_condi.copy() #is that enough?
        if self.noiseRB.isChecked == False:
            self.noiseSB.setDisabled(True)
            return
        else:
            self.noiseSB.setEnabled(True)
            
        # in allowedList traces
        self.noiseCut = self.noiseSB.value()
        print ("Self.noisecut {}".format(self.noiseCut))
        _numberCut = 0
        for _condi in self.tracedata:
            _peaksDF = self.pk_extracted_with_failures[_condi]
            _df = self.tracedata[_condi]
            #print (_df)
            _noise = _df.std()
            #print ("NOISE: ", _noise)
            _peaksDF  =  _peaksDF.where( _peaksDF > _noise * self.noiseCut, 0)
            #print ("peaksdf",_peaksDF )
            _bycol = _peaksDF.isin([0.0]).sum()
            #print ("bycol {} sum {}".format(_bycol, _bycol.sum()))
            _numberCut += _bycol.sum()
            self.pk_extracted_with_failures[_condi] = _peaksDF
        
        # Provide some indication of total peaks altered interactively
        self.peaksLabel.setText("{} peaks set as failures ({: .1f}% of total).".format(_numberCut, 100*_numberCut/self.total_peaks))
        

        
    def maskLowSNR(self):
        """split the peak output according to SNR of parent traces"""
        """moving the spinbox for SNR Cutoff also comes here"""
        print ('maskLowSNR')
        if self.skipRB.isChecked == False:
            self.skipSB.setDisabled(True)
            return
        else:
            self.skipSB.setEnabled(True)
        
        self.excludeSNRcut = self.skipSB.value()

        # use selective region (LR?)
        # or just the whole trace?
        
        #store allowedLists and excludedLists as values with the set as key.
        self.allowedLists = {}
        self.excludedLists = {}
        self.allowedCount = 0
        self.excludedCount = 0
        
        for _condi, _tdf in self.tracedata.items():
            
            #_df = self.tracedata[_condi]
            
            # find SNR from column-wise Max / SD
            _max = _tdf.max()
            _SD = _tdf.std()
            snr = _max / _SD
            #print ("max {}, sd {}, snr {}".format(_max, _SD, snr))
            # add histogram of SNR values with 'SNRcut'-off drawn?
            
            self.allowedLists[_condi] = snr.where(snr >= self.excludeSNRcut).dropna()
            self.excludedLists[_condi] = snr.where(snr < self.excludeSNRcut).dropna()
            
            self.allowedCount += len(self.allowedLists[_condi])
            self.excludedCount += len(self.excludedLists[_condi])
        
            #print ("allowedList: "+_condi, self.allowedLists[_condi])
            #print ("excludedList: "+_condi, self.excludedLists[_condi])
        
        #update dialog
        skipLabelText = "Skipping {0} traces out of {1} for low SNR.".format(self.excludedCount, self.allowedCount + self.excludedCount)
        self.skipRB.setText(skipLabelText)
        
"""

#### TEST CODE ####
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
