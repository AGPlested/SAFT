import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QGroupBox, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pandas as pd

from quantal import fit_nGaussians, nGaussians_display

class testData():
    def __init__(self, *args, **kwargs):
        self.open_file()
        
    def open_file(self):
        _f = "redone13.xlsx"
        #this reads only the first sheet
        
        #need to obtain the sheet names!
        
        #how to get the sheet names from pd.xwx
        self.file = pd.read_excel(_f, index_col=0)
        # need to read all the sheets
        

class HDisplay():
    def __init__(self, *args, **kwargs):
        
        self.plot = pg.GraphicsLayoutWidget()
        self.h = self.plot.addPlot(title="<empty>", row=0, col=0, rowspan=1, colspan=1)
        self.h.setLabel('left', "N")
        self.h.setLabel('bottom', "dF / F")
        self.h.vb.setLimits(xMin=0, yMin=0)
        self.h.addLegend()
        
    def updateTitle(self, newTitle):
        self.h.setTitle (newTitle)

class txOutput():
    def __init__(self, initialText, *args, **kwargs):
        
        self.text = initialText
        self.frame = QtGui.QTextEdit()
        font = QtGui.QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.frame.setCurrentFont(font)
        _w, _h = 300, 600
        self.frame.resize(_w, _h)
        self.frame.setMinimumSize(_w, _h)
        self.frame.setMaximumSize(_w, _h)
        self.appendOutText(initialText)
        self.frame.setReadOnly(True)

    def appendOutText(self, newOP=None):
        if newOP != None:
            self.frame.append(str(newOP))

class histogramFitDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(histogramFitDialog, self).__init__(*args, **kwargs)
        
        #self.peaksScraped = False
        #self.small = 3
        self.outputHeader = "logfile [date]" #fix
        self.hPlot = HDisplay()
        self.outputF = txOutput(self.outputHeader)
        self.makeDialog()
        
        

    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Fit Quantal Histograms")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.resize(1000,500)
        vbox = QVBoxLayout()
        vbox.addWidget(w)
        self.setLayout(vbox)
        
        #histogram view
    
        #work through each ROI in turn - fit summed histogram and then convert to quanta from peaks list
        histograms = QGroupBox("Histogram options")
        histGrid = QGridLayout()
        
        NBin_label = QtGui.QLabel("Number of bins")
        NBin_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_NBin_Spin = pg.SpinBox(value=100, step=10, bounds=[0, 250], delay=0)
        self.histo_NBin_Spin.setFixedSize(80, 25)
        self.histo_NBin_Spin.valueChanged.connect(self.updateHistograms)
        
        histMax_label = QtGui.QLabel("Histogram dF/F max")
        histMax_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_Max_Spin = pg.SpinBox(value=1, step=0.1, bounds=[0.1, 10], delay=0, int=False)
        self.histo_Max_Spin.setFixedSize(80, 25)
        self.histo_Max_Spin.valueChanged.connect(self.updateHistograms)
        
        #toggle show ROI histogram sum
        histsum_label = QtGui.QLabel("Histograms are")
        histsum_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.sum_hist = pg.ComboBox()
        self.sum_hist.setFixedSize(100,25)
        self.sum_hist.addItems(['Separated','Summed'])
        self.sum_hist.currentIndexChanged.connect(self.updateHistograms)
        
        histGrid.addWidget(NBin_label, 2, 0)
        histGrid.addWidget(histMax_label, 1, 0)
        histGrid.addWidget(histsum_label, 0, 0)
        
        
        histGrid.addWidget(self.histo_NBin_Spin, 2, 1)
        histGrid.addWidget(self.histo_Max_Spin, 1, 1)
        histGrid.addWidget(self.sum_hist, 0, 1)
        histograms.setLayout(histGrid)
        
        
        fitParams = QGroupBox("Fitting options")
        fitGrid = QGridLayout()
        #fit parameters
        histnG_label = QtGui.QLabel("Number of Gaussians")
        histnG_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_nG_Spin = pg.SpinBox(value=5, step=1, bounds=[1,10], delay=0, int=True)
        self.histo_nG_Spin.setFixedSize(80, 25)
        self.histo_nG_Spin.valueChanged.connect(self.updateHistograms)
        
        histw_label = QtGui.QLabel("dF_'Q' guess")
        histw_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_q_Spin = pg.SpinBox(value=.05, step=0.01, bounds=[0.01,1], delay=0, int=False)
        self.histo_q_Spin.setFixedSize(80, 25)
        self.histo_q_Spin.valueChanged.connect(self.updateHistograms)
        
        _reFit_label = QtGui.QLabel("with dF_Q fixed")
        _reFitSeparateBtn = QPushButton('ReFit separated')
        _reFitSeparateBtn.clicked.connect(self.reFitSeparated)
        
        fitGrid.addWidget(histnG_label, 0, 0)
        fitGrid.addWidget(self.histo_nG_Spin, 0, 1)
        fitGrid.addWidget(histw_label, 1, 0)
        fitGrid.addWidget(self.histo_q_Spin, 1, 1)
        fitGrid.addWidget(_reFitSeparateBtn, row=2, col=0)
        fitGrid.addWidget(_reFit_label, row=2, col=1)
        fitParams.setLayout(fitGrid)
        
        _doFitBtn = QPushButton('Fit')
        _doFitBtn.clicked.connect(self.fitGaussians)
        layout.addWidget(_doFitBtn, 3, 1)
        
        _skipBtn = QPushButton('Skip')
        _skipBtn.clicked.connect(self.skipROI)
        layout.addWidget(_skipBtn, row=2, col=1)
        
        _saveBtn = QPushButton('Save')
        _saveBtn.clicked.connect(self.save)
        layout.addWidget(_saveBtn, row=1, col=1)
        
        layout.addWidget(histograms, 1 , 1, 1, 1)
        layout.addWidget(fitParams, 1, 0, 1, 1)
        layout.addWidget(self.hPlot.plot, 0, 0, 1, 2)
        layout.addWidget(self.outputF.frame, 0, 3, -1, 1)
        
        self.setLayout(layout)
    
    def histogram_parameters(self):
        _nbins = int(self.histo_NBin_Spin.value())
        _max = self.histo_Max_Spin.value()
        self.outputF.appendOutText ("N_bins {}, Max {}".format(_nbins, _max))
        return _nbins, _max
    
    def skipROI(self):
        self.outputF.appendOutText ("move to the next ROI")
    
    def save(self):
        self.outputF.appendOutText ("write data out")
    
    def reFitSeparated(self):
        """obtain Pr using binomial, using q and w from previous fit"""
        
        
        self.outputF.appendOutText ("refit seperate histograms with fixed q and binomial")
        
        #use last df_Q fit
        
        #need to subclass multiple gaussians to include Pr
        
            
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
        
        """
        #True if box is checked, otherwise False
        self.ignore =  self.skipRB.isChecked()
        
        self.psr = self.psrSB.value() // 2          #floor division to get ears
        """
    
    def fitGaussians(self):
        self.updateHistograms()
    
        
    def addData(self, data):
        """Bring in external data for analysis"""
        
        #take all peak lists
        
        
        self.peakResults = data
        #print (self.peakResults)
        
        #not getting multiple sheets yet - need to fix input
        tdk = self.peakResults.keys()
        tdk_display = ", ".join(str(k) for k in tdk)
        N_ROI = [len (self.peakResults[d].columns) for d in tdk]
        
        self.N_ROI_label.setText("Scraping peaks from {} ROIs \n over the sets named {}".format(N_ROI, tdk_display))
        
        _printable = "{}\n{}\n".format(tdk_display, [self.tracedata[d].head() for d in tdk])
        self.outputF.appendOutText ("Added data of type {}:\n{}\n".format(type(self.tracedata), _printable))
        
       
        
    def updateHistograms(self):
        """called when histogram controls are changed"""
           
        # get controls values and summarise to terminal
        _nbins, _max = self.histogram_parameters()
        #_ROI = self.ROI_selectBox.currentText()
        _hsum = self.sum_hist.currentText()
        print ('Update {3} Histogram(s) for {2} with Nbins = {0} and maximum dF/F = {1}.'.format(_nbins, _max, _ROI, _hsum))
          
        # clear
        self.h.clear()
           
        if _hsum == "Separated":
            for i, _set in enumerate(self.sheets):
                # colours
                col_series = (i, len(self.sheets))
                # get relevant peaks data for displayed histograms
                _, _pdata = self.peakResults.getPeaks(_ROI, _set)
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                # replot
                self.h.plot(hx, hy, name="Histogram "+_set, stepMode=True, fillLevel=0, fillOutline=True, brush=col_series)

        elif _hsum == "Summed":
            sumhy = np.zeros(_nbins)
            for _set in self.sheets:
                _, _pdata = self.peakResults.getPeaks(_ROI, _set)
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                sumhy += hy
           
            self.p2.plot(hx, sumhy, name="Summed histogram "+_ROI, stepMode=True, fillLevel=0, fillOutline=True, brush='y')
           
            if self.fitHistogramsOption:
                print ("lens hx, hy", len(hx), len(hy))
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                _ws = self.histo_Max_Spin.value() / 20
               
                _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                opti = fit_nGaussians(_num, _q, _ws, sumhy, _hxc)
                _hx_u, _hy_u = nGaussians_display (_hxc, _num, opti)
                _qfit = opti.x[0]
                _c = self.p2.plot(_hx_u, _hy_u, name='Fit of {} Gaussians q: {:.2f}'.format(_num,_qfit))
                #from pyqtgraph.examples
                _c.setPen('w', width=3)
                _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))




        
        
if __name__ == '__main__':
    
    tdata = testData()
    tdata.open_file()
    
    app = QApplication([])
    main_window = histogramFitDialog()
    #main_window.addData(tdata.file)
    main_window.show()
    sys.exit(app.exec_())
