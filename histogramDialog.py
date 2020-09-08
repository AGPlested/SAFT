import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QGridLayout, QGroupBox, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pandas as pd

from quantal import fit_nGaussians, nGaussians_display, fit_nprGaussians, nprGaussians_display

def despace (s):
    sp = " "
    if sp in s:
        return s.split(' ')[0]
    else:
        return s

class testData():
    def __init__(self, *args, **kwargs):
        self.open_file()
        
    def open_file(self):
        _f = "redone13.xlsx"
      
        #"None" reads all the sheets into a dictionary of data frames
        self.histo_df = pd.read_excel(_f, None, index_col=0)
        #print (self.file_dict)

class ROI_Controls(QtGui.QWidget):
    #from Stack Overflow : https://stackoverflow.com/questions/56267781/how-to-make-double-arrow-button-in-pyqt
    #and https://programming.vip/docs/pyqt5-note-7-multiple-class-sharing-signals.html
    ctrl_signal = QtCore.Signal()
    def __init__(self, parent, *args):
        super(ROI_Controls, self).__init__(*args)
        
        self.parent = parent
        self.ROI_box = QGroupBox("ROI")
        l = QGridLayout()
        
        buttonLL = QtGui.QToolButton()
        buttonLL.setIcon(buttonLL.style().standardIcon(QtGui.QStyle.SP_MediaSeekBackward))

        buttonL = QtGui.QToolButton()
        buttonL.setArrowType(QtCore.Qt.LeftArrow)

        buttonR = QtGui.QToolButton()
        buttonR.setArrowType(QtCore.Qt.RightArrow)

        buttonRR = QtGui.QToolButton()
        buttonRR.setIcon(buttonRR.style().standardIcon(QtGui.QStyle.SP_MediaSeekForward))

        #lay = QtGui.QHBoxLayout(self)
        buttonList = [buttonLL, buttonL, buttonR, buttonRR]
        for counter, btn in enumerate(buttonList):
            print (counter, btn)
            
            btn.pressed.connect(lambda val=counter: self.buttonPressed(val))
            #btn.clicked.connect(self.ctrl_signal.emit)
            #self.ctrl_signal.connect(parent.ctrl_signal.emit)
            l.addWidget(btn, 0, counter)
          
        self.ROI_label = QtGui.QLabel("-")
        l.addWidget(self.ROI_label, 0, 4)
        self.ROI_box.setLayout(l)
        
        #self.ctrl_signal.connect(self.buttonPressed)

    def update_ROI_label(self, t):
        self.ROI_label.setText (t)

    #@pyqtSlot(int)
    def buttonPressed(self, _b):
        print (_b)
        self.ROI_label.setText(str(_b))
        self.parent.ROI_change_command (_b)
        
    
    
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
    
    ctrl_signal = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super(histogramFitDialog, self).__init__(*args, **kwargs)
        
        self.fitHistogramsOption = False
        
        self.outputHeader = "logfile [date]" #fix
        self.hPlot = HDisplay()
        self.outputF = txOutput(self.outputHeader)
        self.makeDialog()
    
    def test(self, sender):
        print (sender)
        self.outputF.appendOutText ('ctrl_button was pressed {}'.format(sender))
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        self.setWindowTitle("Fit Quantal Histograms")
        layout = QGridLayout()
        w = QWidget()
        w.setLayout(layout)
        
        self.RC = ROI_Controls(self)        #need to send this instance as parent
        #self.ctrl_signal.connect(self.test)
        
        self.resize(1000,500)
        gbox = QGridLayout()
        gbox.addWidget(self.RC.ROI_box,0,0,1,1)
        gbox.addWidget(w,1,0,1,-1)
        self.setLayout(gbox)
        
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
        layout.addWidget(_doFitBtn, 3, 0)
        
        _saveAdvBtn = QPushButton('Store Fits and Advance')
        _doFitBtn.clicked.connect(self.save)
        layout.addWidget(_saveAdvBtn, 3, 1)
        
        _skipBtn = QPushButton('Skip ROI')
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
        
        self.fitHistogramsOption = True
        
        self.outputF.appendOutText ("refit seperate histograms with fixed q and binomial")
        
        #use last df_Q fit
        _q = self.fixq
        #need to subclass multiple gaussians to include Pr
        self.updateHistograms()
            
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
        _hsum = self.sum_hist.currentText()
        if _hsum == "Separated":
            pass
        elif _hsum == "Summed":
            self.fitHistogramsOption = True
        self.updateHistograms()
        
        
    def addData(self, data):
        """Bring in external data for analysis"""
        
        #take all peak lists
    
        self.peakResults = data # a dict of DataFrames
        #print (self.peakResults)
        
        #the histograms aren't much use, we might change binning
        if 'histograms' in self.peakResults:
            del self.peakResults['histograms']
        
        #clean out any low SNR data _ avoid interating over ordered dict
        for key in self.peakResults.copy():
            if "SNR<" in key:
                del self.peakResults[key]
        
        tdk = self.peakResults.keys()
        tdk_display = ", ".join(str(k) for k in tdk)
        print (tdk_display)
        N_ROI = [len (self.peakResults[d].columns) for d in tdk]
        
        self.outputF.appendOutText ("Scraping peaks from {} ROIs \n over the sets named {}".format(N_ROI, tdk_display))
        
        _printable = "{}\n{}\n".format(tdk_display, [self.peakResults[d].head() for d in tdk])
        self.outputF.appendOutText ("Added data of type {}:\n{}\n".format(type(self.peakResults), _printable))
        
        
        #self.ROI_list = list(k.split()[0] for k in self.ROI_list)
        
        for d in self.peakResults:
            self.peakResults[d].rename(despace, axis='columns', inplace=True)
            
        self.ROI_list = list(self.peakResults[list(tdk)[0]].keys().unique(level=0))
        print (self.ROI_list)
        
        self.ROI_change()   #default is the first (0).
        self.updateHistograms()
    
    def ROI_change_command (self, button_command):
        print(button_command)
        if button_command == 0:
            self.ROI_N = 0
        elif button_command == 1:
            self.ROI_N -= 1
            if self.ROI_N < 0:  self.ROI_N = 0
        elif button_command == 2:
            self.ROI_N += 1
            if self.ROI_N == len(self.ROI_list):  self.ROI_N = len(self.ROI_list) - 1
        elif button_command == 3:
            self.ROI_N = len(self.ROI_list) - 1
        
        print ("self ROI N:", self.ROI_N)
        self.ROI_change(self.ROI_N)
        
    def ROI_change(self, _ROI=0):
        self.ROI_N = _ROI
        self.current_ROI = self.ROI_list[_ROI]
        self.RC.update_ROI_label("{} : {} of {}".format(self.current_ROI, self.ROI_N + 1, len(self.ROI_list)))
        
        #reset fits before calling!
        self.updateHistograms()
        
        
    def updateHistograms(self):
        """called when histogram controls are changed"""
           
        # get values from controls and summarise to terminal
        _nbins, _max = self.histogram_parameters()
        _ROI = self.current_ROI
        _hsum = self.sum_hist.currentText()
        print ('Update {3} Histogram(s) for {2} with Nbins = {0} and maximum dF/F = {1}.'.format(_nbins, _max, _ROI, _hsum))
          
        # clear
        self.hPlot.h.clear()
           
        #need them split to see
    
        if _hsum == "Separated":
            for i, _set in enumerate(self.peakResults.keys()):
                # colours
                col_series = (i, len(self.peakResults.keys()))
                # get relevant peaks data for displayed histograms
                _pdata = self.peakResults[_set][_ROI]
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                # replot
                self.hPlot.h.plot(hx, hy, name="Histogram "+_set, stepMode=True, fillLevel=0, fillOutline=True, brush=col_series)
            
                if self.fitHistogramsOption:
                    #binomial path
                    _q = self.fixq
                    _num = self.histo_nG_Spin.value()
                    _ws = self.fixws
                    _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                    opti = fit_nprGaussians (_num, _q, _ws, hy, _hxc)
                    _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, opti)
                    _pr = opti.x[0]
                    print (opti)
                    _c = self.hPlot.h.plot(_hx_u, _hy_u, name='Fit of {} Gaussians, Pr: {:.2f}, q: {:.2f}'.format(_num,_pr,_q))
                    #write out Pr in log!
                    #from pyqtgraph.examples
                    _c.setPen('w', width=3)
                    _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                        
        elif _hsum == "Summed":
            sumhy = np.zeros(_nbins)
            for _set in self.peakResults.keys():
                _pdata = self.peakResults[_set][_ROI]
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                sumhy += hy
           
            self.hPlot.h.plot(hx, sumhy, name="Summed histogram "+_ROI, stepMode=True, fillLevel=0, fillOutline=True, brush='y')
           
            if self.fitHistogramsOption:
                print ("lens hx, hy", len(hx), len(hy))
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                _ws = self.histo_Max_Spin.value() / 20
               
                _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                opti = fit_nGaussians(_num, _q, _ws, sumhy, _hxc)
                _hx_u, _hy_u = nGaussians_display (_hxc, _num, opti)
                _qfit = opti.x[0]
                _wsfit = opti.x[1]
                _c = self.hPlot.h.plot(_hx_u, _hy_u, name='Fit of {} Gaussians q: {:.2f}, w: {:.2f}'.format(_num, _qfit, _wsfit))
                self.fixq = _qfit
                self.fixws = _wsfit
                #from pyqtgraph.examples
                _c.setPen('w', width=3)
                _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))




        
        
if __name__ == '__main__':
    
    tdata = testData()
    tdata.open_file()
    
    app = QApplication([])
    main_window = histogramFitDialog()
    main_window.addData(tdata.histo_df)
    main_window.show()
    sys.exit(app.exec_())
