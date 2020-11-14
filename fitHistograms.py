import sys
import datetime
import os.path
import copy

### Package imports
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QFileDialog, QApplication, QMainWindow, QGridLayout, QGroupBox, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout, QCheckBox
import numpy as np
import pyqtgraph as pg
import pandas as pd
from scipy.stats import chisquare, kstest

### SAFT imports
from dataStructures import HistogramFitStore as HFStore
from dataStructures import histogramFitParams
from utils import getRandomString, linePrint
from quantal import fit_nGaussians, nGaussians_display, fit_nprGaussians, fit_PoissonGaussians_global, PoissonGaussians_display, nprGaussians_display, fit_nprGaussians_global, nprGaussians, poissonGaussians, cdf

def despace (s):
    sp = " "
    if sp in s:
        return s.split(' ')[0]
    else:
        return s

class testData():
    # if test data is loaded automatically, you can't load any more data
    def __init__(self, *args, **kwargs):
        self.open_file()
        
    def open_file(self):
        self.filename = "ExPeak_Data.xlsx"
      
        #"None" reads all the sheets into a dictionary of data frames
        self.histo_df = pd.read_excel(self.filename, None, index_col=0)
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
        
        buttonList = [buttonLL, buttonL, buttonR, buttonRR]
        bsize = (75, 20)
        for b in buttonList:
            b.setFixedSize(*bsize)
        
        _clearFitsBtn = QPushButton('Clear recent fits')
        buttonList.append(_clearFitsBtn)
        _clearFitsBtn.setFixedWidth(220)
        
        _storeAdvBtn = QPushButton('Next ROI, keep fits')
        buttonList.append(_storeAdvBtn)
        _storeAdvBtn.setFixedWidth(220)
        
        _skipBtn = QPushButton('Next ROI, discard fits')
        buttonList.append(_skipBtn)
        _skipBtn.setFixedWidth(220)

        posn = [(0,2), (0,3), (0,4), (0,5), (1,0,1,2), (1,2,1,2), (1,4,1,2)]
        
        for counter, btn in enumerate(buttonList):
            
            btn.pressed.connect(lambda val=counter: self.buttonPressed(val))
           
            #self.ctrl_signal.connect(parent.ctrl_signal.emit)
            l.addWidget(btn, *posn[counter])
          
        self.ROI_label = QtGui.QLabel("-")
        self.ROI_label.setFixedSize(150, 40)
        self.ROI_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        l.addWidget(self.ROI_label, 0, 0, 2, 1)
        self.ROI_box.setLayout(l)

    def update_ROI_label(self, t):
        self.ROI_label.setText (t)
        #with pyside2 5.13 this label (and other GUI items) doesn't update
        
    def buttonPressed(self, _b):
        #print (_b)
        self.parent.ROI_change_command (_b)
        
    
    
class HDisplay():
    def __init__(self, *args, **kwargs):
        
        self.glw = pg.GraphicsLayoutWidget()
        self.hrc = (0, 0, 1, 1) #same as below
        self.h = self.glw.addPlot(title="<empty>", *self.hrc)
        self.h.setLabel('left', "N")
        self.h.setLabel('bottom', "dF / F")
        self.h.vb.setLimits(xMin=0, yMin=0)
        self.h.addLegend(offset=(-10, 5))
        self.stack = pg.GraphicsLayout()
        
          
    def updateTitle(self, newTitle):
        self.h.setTitle (newTitle)
        
    def createSplitHistLayout(self, keys):
        """separated view with each Histogram stacked in separate plots"""
        # adapted from SAFT
        # Store the plot items in a list - can't seem to get them easily otherwise?
        data = []   # empty
        self.stackMembers = []
        for _condition in keys:
        
            memberName = _condition + " histogram"
            stack_member = self.stack.addPlot(y=data, name=memberName)
            
            stack_member.vb.setLimits(xMin=0, yMin=0)
            stack_member.hideAxis('bottom')
            stack_member.addLegend(offset=(-10, 5))
            stack_member.setLabel('left', "N")
            self.stackMembers.append(stack_member)
            self.stack.nextRow()
            #print (c, len(self.p1stackMembers))
        
        #link y-axes - using final member of stack as anchor
        for s in self.stackMembers:
            if s != stack_member:
                s.setXLink(stack_member)
                s.setYLink(stack_member)
                
        #add back bottom axis to the last graph
        stack_member.showAxis("bottom")
        stack_member.setLabel('bottom', "dF / F")

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
    
class histogramFitDialog(QDialog):
    
    ctrl_signal = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super(histogramFitDialog, self).__init__(*args, **kwargs)
        _now = datetime.datetime.now()
        
        #can be set to "None" following a long fit routine to stop cycling
        self.fitHistogramsOption = "Summed"
        self.outputHeader = "{} Logfile\n".format(_now.strftime("%y%m%d-%H%M%S"))
        self.hPlot = HDisplay()
        self.saveFits = False
        self.filename = None
        self.dataname = None
        self.fitPColumns = ['ROI', 'Fit ID', 'N', 'Pr/mu', 'Events', 'Width', 'dF_Q', 'Test', 'Stat.', 'P_val', 'Type']
        self.fitInfoHeader = "Fits for current ROI:\n" + linePrint(self.fitPColumns)
        self.outputF = txOutput(self.outputHeader)
        self.current_ROI = None
        self.flag = "Auto max"      #how to find histogram x-axis
        self.makeDialog()
    
    """
    def test(self, sender):
        print (sender)
        self.outputF.appendOutText ('ctrl_button was pressed {}'.format(sender))
    """
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        # work through each ROI in turn - fit summed histogram and then convert to quanta from peaks list
        
        self.setWindowTitle("Fit Histograms with Quantal Parameters")
        #self.resize(1000, 800)
        
        # panel for file commands and information
        _fileOptions = QGroupBox("File")
        _fileGrid = QGridLayout()
        
        self.loadBtn = QPushButton('Load')
        self.loadBtn.clicked.connect(self.openData)
        self.loadBtn.setDisabled(True)
        self.saveBtn = QPushButton('Save')
        self.saveBtn.clicked.connect(self.save)
        self.saveBtn.setDisabled(True)
        self.dataname_label = QtGui.QLabel("No data")
        self.sHFCurves = QCheckBox('Save Histogram Fitted Curves')
        self.sHFCurves.setChecked(False)
        self.sHFCurves.stateChanged.connect(self.toggleSaveFits)
        
        _fileGrid.addWidget(self.loadBtn, 0, 0, 1, 1)
        _fileGrid.addWidget(self.saveBtn, 1, 0, 1, 1)
        _fileGrid.addWidget(self.sHFCurves, 3, 0, 1, 1)
        _fileGrid.addWidget(self.dataname_label, 2, 0, 1, 1)
        _fileOptions.setLayout(_fileGrid)
        
        # panel of display options for histograms
        _histOptions = QGroupBox("Histogram options")
        _histGrid = QGridLayout()
        
        _NBin_label = QtGui.QLabel("No. of bins")
        _NBin_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_NBin_Spin = pg.SpinBox(value=100, step=10, bounds=[0, 250], delay=0)
        self.histo_NBin_Spin.setFixedSize(80, 25)
        self.histo_NBin_Spin.valueChanged.connect(self.updateHistograms)
        
        _histMax_label = QtGui.QLabel("Max dF/F")
        _histMax_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_Max_Spin = pg.SpinBox(value=1, step=0.1, bounds=[0.1, 10], delay=0, int=False)
        self.histo_Max_Spin.setFixedSize(80, 25)
        self.histo_Max_Spin.valueChanged.connect(self.setManualMax)
        
        # toggle show ROI histogram sum
        _histsum_label = QtGui.QLabel("Display histograms")
        _histsum_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.sum_hist = pg.ComboBox()
        self.sum_hist.setFixedSize(100,25)
        self.sum_hist.addItems(['Summed', 'Separated'])
        self.split_state = False #because summed is first in list and default
        self.sum_hist.currentIndexChanged.connect(self.updateHistograms)
        
        _histGrid.addWidget(_NBin_label, 2, 0)
        _histGrid.addWidget(_histMax_label, 1, 0)
        _histGrid.addWidget(_histsum_label, 0, 0)
        _histGrid.addWidget(self.histo_NBin_Spin, 2, 1)
        _histGrid.addWidget(self.histo_Max_Spin, 1, 1)
        _histGrid.addWidget(self.sum_hist, 0, 1)
        _histOptions.setLayout(_histGrid)
        
        # panel of fit parameters and commands
        _fittingPanel = QGroupBox("Fitting")
        _fitGrid = QGridLayout()

        _histnG_label = QtGui.QLabel("Gaussian components")
        _histnG_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_nG_Spin = pg.SpinBox(value=6, step=1, bounds=[1,20], delay=0, int=True)
        self.histo_nG_Spin.setFixedSize(50, 25)
        self.histo_nG_Spin.setAlignment(QtCore.Qt.AlignRight)
        #print (self.histo_nG_Spin.alignment())
        self.histo_nG_Spin.valueChanged.connect(self.updateHistograms)
        
        _histw_label = QtGui.QLabel("dF (q) guess")
        _histw_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_q_Spin = pg.SpinBox(value=.05, step=0.005, bounds=[0.005,1], delay=0, int=False)
        self.histo_q_Spin.setFixedSize(80, 25)
        self.histo_q_Spin.setAlignment(QtCore.Qt.AlignRight)
        self.histo_q_Spin.valueChanged.connect(self.updateHistograms)
        
        _reFit_label = QtGui.QLabel("fixed dF (q), w")
        self.reFitSeparateBtn = QPushButton('Separate Binomial fits')
        self.reFitSeparateBtn.clicked.connect(self.reFitSeparated)
        self.reFitSeparateBtn.setDisabled(True)
        
        _sumFit_label = QtGui.QLabel("free dF (q), w, amplitudes")
        _doFitBtn = QPushButton('Fit Summed')
        _doFitBtn.clicked.connect(self.fitGaussians)
        
        _globFit_label = QtGui.QLabel("find N, Pr; common dF (q), w")
        _globFitBtn = QPushButton('Global Binomial fit')
        _globFitBtn.clicked.connect(self.fitGlobalGaussians)
        
        _PoissonGlobalFit_label = QtGui.QLabel("find mu; common dF (q), w")
        _PoissonGlobalFitBtn = QPushButton('Global Poisson fit')
        _PoissonGlobalFitBtn.clicked.connect(self.PoissonfitGlobalGaussians)
        
        self.fitInfo = txOutput(self.fitInfoHeader)
        self.fitInfo.size(500, 120)
        
        
        _fitGrid.addWidget(_histnG_label, 0, 2)
        _fitGrid.addWidget(self.histo_nG_Spin, 0, 3)
        _fitGrid.addWidget(_histw_label, 0, 0)
        _fitGrid.addWidget(self.histo_q_Spin, 0, 1)
        _fitGrid.addWidget(_doFitBtn, 1, 0, 1, 2)
        _fitGrid.addWidget(_sumFit_label, 1, 2, 1, 3)
        _fitGrid.addWidget(self.reFitSeparateBtn, 2, 0, 1, 2)
        _fitGrid.addWidget(_reFit_label, 2, 2, 1, 3)
        _fitGrid.addWidget(_globFitBtn, 3, 0, 1, 2)
        _fitGrid.addWidget(_globFit_label, 3, 2, 1, 3)
        _fitGrid.addWidget(_PoissonGlobalFitBtn, 4, 0, 1, 2)
        _fitGrid.addWidget(_PoissonGlobalFit_label, 4, 2, 1, 3)
        _fitGrid.addWidget(self.fitInfo.frame, 0, 5, -1, 1)
        _fittingPanel.setLayout(_fitGrid)
        
        # histogram analysis layout
        self.hlayout = QGridLayout()
        
        # histogram view
        self.histogramLayPos = (0, 0, 2, 3)
        self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
        
        _fittingPanel.setFixedHeight(180)
        self.hlayout.addWidget(_fittingPanel, 3, 0, 1, 3)
        
        # ROI controls
        self.RC = ROI_Controls(self)        #need to send this instance as parent
        self.RC.ROI_box.setFixedHeight(100)
        self.hlayout.addWidget(self.RC.ROI_box, 2, 0, 1, 2)
        
        # Display options for the histograms
        _histOptions.setFixedSize(250, 100)
        self.hlayout.addWidget(_histOptions, 2, 2, 1, 1)
        
        # Text output console
        self.outputF.frame.setFixedSize(300, 550)
        self.hlayout.addWidget(self.outputF.frame, 0, 3, 2, 1)
        
        _fileOptions.setFixedSize(300, 200)
        self.hlayout.addWidget(_fileOptions, 2, 3, 3, 1)
        
        self.setLayout(self.hlayout)
    
    def setManualMax(self):
        self.flag = "Manual max"
        self.updateHistograms()
    
    def histogram_parameters(self):
        _nbins = int(self.histo_NBin_Spin.value())
        _max = self.histo_Max_Spin.value()
        self.outputF.appendOutText ("N_bins {}, Max {}".format(_nbins, _max))
        return _nbins, _max
    
    def clearFits(self):
    
        self.outputF.appendOutText ("Discarded fit results from {}".format(self.current_ROI), "red")
        
        # clear current fits and info frame
        self.currentROIFits = pd.DataFrame(columns=self.currentROIFits.columns)
        
        self.fitInfo.reset(self.fitInfoHeader)
    
    def skipROI(self):
        
        self.clearFits()
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI), "magenta")
        
    def toggleSaveFits(self):
        if self.sHFCurves.isChecked == False:
            self.saveFits = False
    
        else:
            self.saveFits = True
        
        
    def save(self):
        #maybe we just have a filename not a path
        self.outputF.appendOutText ("Keeping {} fit results for {} --\n".format(len(self.currentROIFits.index),self.current_ROI), "Magenta")
        
        self.fitResults = self.fitResults.append(copy.copy(self.currentROIFits), ignore_index=True)
        
        if self.filename != None:
        
            if os.path.split(self.filename)[0] is not None:
                _outfile = os.path.split(self.filename)[0] + "HFit_" + os.path.split(self.filename)[1]
                
            else :
                _outfile = "HFit_" + self.filename
        
        else:
            _outfile = self.dataname + "_HFit.xlsx"
           
        with pd.ExcelWriter(_outfile) as writer:
            
            self.fitResults.to_excel(writer, sheet_name="Fit Results", startrow=1)
        
            #optionally all fitted curves are saved
            if self.saveFits:
        
                #repack as dict of dataframes
                fits_ddf = {}
                for k,v in self.Fits_data.items():
                    fits_ddf[k] = v.df
            
                _fitsDF = pd.concat(fits_ddf, axis=1)
                _fitsDF.columns.rename(["Fit ID", "Condition", "Coordinate"], inplace=True )
                #print (_fitsDF.head(5))
        
                _fitsDF.to_excel(writer, sheet_name="Histograms & Fitted Curves", startrow=1)
        
        print ("Wrote {} to disk.".format(_outfile))
        self.outputF.appendOutText ("Wrote fit results out to disk: {}".format(_outfile))
       

    
    def storeAdvance(self):
        """storing data and moving forward one ROI"""
        self.outputF.appendOutText ("Keeping {} fit results for {} --\n".format(len(self.currentROIFits.index),self.current_ROI), "Magenta")
        
        self.fitResults = self.fitResults.append(copy.copy(self.currentROIFits), ignore_index=True)
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI), "Magenta")
        # empty the current fits dataframe
        self.currentROIFits = pd.DataFrame(columns=self.currentROIFits.columns)
        self.fitInfo.reset(self.fitInfoHeader)
        
    
    def reFitSeparated(self):
        """obtain Pr using binomial, using q and w from previous fit"""
        
        # use last df_Q from last summed fit
        _q = self.fixq
        self.outputF.appendOutText ("Refit separate histograms ({}) with fixed q {:.3f} and binomial Pr.".format(self.current_ROI, _q))
        
        self.fitHistogramsOption = "Individual"
        self.sum_hist.setCurrentIndex(1)    #calls updateHistogram and fit
    
    def PoissonfitGlobalGaussians(self):
        """obtain mean release rate using Poisson, estimating q, w and scale from global fit"""
        self.fitHistogramsOption = "Global Poisson"
        self.outputF.appendOutText ("Global Poisson Fit")
        _fitSum =self.sum_hist.currentIndex()
        if _fitSum == 1:
            self.updateHistograms()             # no need to adjust view (histograms already separate), just fit
        else:
            self.sum_hist.setCurrentIndex(1)    # sets view to separate histograms, calls update histograms and performs the fit.
            
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
        pass
        
    
    def fitGlobalGaussians(self):
        """Target of the global fit button"""
        self.fitHistogramsOption = "Global Binom"
        self.outputF.appendOutText ("Global Binomial Fit", "Magenta")
        _fitSum =self.sum_hist.currentIndex()
        if _fitSum == 1:
            self.updateHistograms()             # no need to adjust view (histograms already separate), just fit
        else:
            self.sum_hist.setCurrentIndex(1)    # sets view to separate histograms, calls update histograms and performs the fit.
        
    def fitGaussians(self):
        """Target of the fit summed button"""
        
        self.fitHistogramsOption = "Summed"
        _fitSum =self.sum_hist.currentIndex()
        #self.split_state == False
        
        if _fitSum == 0:
            self.updateHistograms()             # no need to adjust view, just update and fit
        else:
            self.sum_hist.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.

    def openData(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open Data", os.path.expanduser("~"))[0]
        
        if self.filename:
            #"None" reads all the sheets into a dictionary of data frames
            self.open_df = pd.read_excel(self.filename, None, index_col=0)
            self.addData (self.open_df)
            self.outputF.appendOutText ("Opening file {}".format(self.filename))
        
    def addData(self, _data, _name=None):
        """
        Bring in external data for analysis
        _data should be a dictionary of dataframes
        keys are conditions
        values are dataframes with times as indices (ignored)
        and peak amplitudes as columns named by ROIs
        """
        
        if self.filename:
            #show only the filename not the entire path
            _f = os.path.split(self.filename)[1]
            self.dataname_label.setText (_f)
        
        if _name:
            self.dataname_label.setText (_name)
            self.dataname = _name
        
        #take all peak lists
        self.peakResults = _data # a dict of DataFrames

        
        # any histograms aren't much use, we will change binning, so remove them
        if 'histograms' in self.peakResults:
            del self.peakResults['histograms']
        
        # clean out any low SNR data _ avoid interating over an ordered dict
        for key in self.peakResults.copy():
            if "SNR<" in key:
                del self.peakResults[key]
                self.outputF.appendOutText ("Removed low SNR data {}".format(key),"Red")
        
        pRK = self.peakResults.keys()
        pRK_display = ", ".join(str(k) for k in pRK)
        print (pRK_display)
        N_ROI = [len (self.peakResults[d].columns) for d in pRK]
        
        #self.outputF.appendOutText ("Scraping peaks from {} ROIs \n over the sets named {}".format(N_ROI, pRK_display))
        
        self.outputF.appendOutText ("Added data of type {}:\n".format(type(self.peakResults)))
        for _c, _d in zip(pRK, [self.peakResults[d].head() for d in pRK]):
            self.outputF.appendOutText ("{}\n{}\n".format(_c, _d), "Blue")
        
        for d in self.peakResults:
            self.peakResults[d].rename(despace, axis='columns', inplace=True)
           
        # minimal approach, consider exhaustive list as in SAFT.py
        self.ROI_list = list(self.peakResults[list(pRK)[0]].keys().unique(level=0))
        print (self.ROI_list)
        
        
       
        
        self.fitResults = histogramFitParams(pRK, self.fitPColumns)
        self.currentROIFits = histogramFitParams(pRK, self.fitPColumns)
        self.Fits_data = {}
        
        self.hPlot.createSplitHistLayout(self.peakResults.keys())
        
        self.ROI_change()   # default is the first (0).
        self.updateHistograms()
    
    def ROI_change_command (self, button_command):
        #print("Button command: {}".format(button_command))
        
        # turn off separate fitting when moving to new ROI, and get histogram x-range automatically
        self.reFitSeparateBtn.setDisabled(True)
        self.flag = "Auto max"
        
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
        
        elif button_command == 4:
            self.clearFits()
            return
            
        elif button_command == 5:
            self.storeAdvance() # comes back here with b_c = 2
            return
        
        elif button_command == 6:
            self.skipROI()  # comes back here with b_c = 2
            return
            
        print ("self_ROI_N is ", self.ROI_N)
        self.ROI_change(self.ROI_N)
        
    def ROI_change(self, _ROI=0):
        """ update on ROI change"""
        self.ROI_N = _ROI
        self.current_ROI = self.ROI_list[_ROI]
        self.RC.update_ROI_label("{} : {} of {}".format(self.current_ROI, self.ROI_N + 1, len(self.ROI_list)))
        
        #for any change of ROI, the default view is the sum of all histograms
        _fitSum =self.sum_hist.currentIndex()
        self.fitHistogramsOption = "Summed"
        
        if _fitSum == 0:
            self.updateHistograms()
            # if the view was already "summed", no need to adjust, just update and fit
        else:
            self.sum_hist.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.
        
    def updateHistograms(self):
        """Histogram controls were changed, redo the fit.
        """
        
        # would be None if no data loaded so escape
        if self.current_ROI == None:
            return
        
        _ROI = self.current_ROI
        # get values from controls and summarise them to terminal
        if self.flag == "Manual max":
            _nbins, _max = self.histogram_parameters()
            #print ("Manual max triggered {} {}".format(_nbins, _max))
        else:
            #auto max of histogram
            #print("self.flag is {}".format(self.flag))
            _nbins  = self.histogram_parameters()[0]
            _max = 0
            for _condition in self.peakResults.keys():
                _pdata = self.peakResults[_condition][_ROI]
                if _pdata.max() > _max * 1.2:
                    _max = _pdata.max() * 1.2
            
        
        _hsum = self.sum_hist.currentText()
        self.outputF.appendOutText ("Update {0} Histogram(s) for {1} with {2} bins and maximum dF/F = {3:0.3f}.".format(_hsum, _ROI, _nbins, _max))
    
        if "Global" in self.fitHistogramsOption:
            ystack = []
            xstack = []
        
        # _ID is a unique identifier for each fit
        # it is the key for the fits dict
        _ID = getRandomString(4)
        
        if self.fitHistogramsOption not in ["None", "Individual", "Summed"]:
            
            self.Fits_data[_ID] = HFStore(self.current_ROI, self.peakResults.keys())
            #print ("SFHO: {}\nSFDID: {}".format(self.fitHistogramsOption, self.Fits_data[_ID]))
            
        if _hsum == "Separated":
            _num = self.histo_nG_Spin.value()
            #_pr_results = [_num]
            
            if self.split_state == False:
                # bring in split histogram view
                # remove single
                self.hPlot.glw.removeItem(self.hPlot.h)
                # add multiple
                self.hPlot.glw.addItem(self.hPlot.stack)
                
                ####### replace in layout *4-tuple has position and size NOT NEEDED
                #######self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
                
                # when complete, toggle to bypass next time
                self.split_state = True
            
            # Get the next empty line in the dataframe for writing results out
            imax = self.currentROIFits.index.max()
            if np.isnan(imax):
                imax = 0
            
            _ymax = 0
            for i, _condition in enumerate(self.peakResults.keys()):
                # colours
                col_series = (i, len(self.peakResults.keys()))
                # get relevant peaks data for displayed histograms
                _pdata = self.peakResults[_condition][_ROI]
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                
                if hy.max() > _ymax:
                    _ymax = hy.max()
                # Only store histogram values if we are doing a global fit
                if self.fitHistogramsOption not in ["None", "Individual", "Summed"]:
                    print (self.Fits_data)
                    self.Fits_data[_ID].addHData(_condition, hx, hy)
                
                if "Global" in self.fitHistogramsOption:
                    ystack.append(hy)
                    xstack.append(hx)
                
                # replot in the right place in the stack
                target = self.hPlot.stackMembers[i]
                target.clear()   # this unfortunately cleans out any text - we should instead remove the hist + fit?
                target.plot(hx, hy, name="{} {} Histogram".format(_ROI, _condition), stepMode=True, fillLevel=0, fillOutline=True, pen=col_series, brush=col_series)
            
                if self.fitHistogramsOption == "Individual":
                    # binomial path
                    _q = self.fixq
                    _num = self.histo_nG_Spin.value()
                    _ws = self.fixws # / 1.5              # arbitrary but always too fat!
                    _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                    _opti = fit_nprGaussians (_num, _q, _ws, hy, _hxc)
                    
                    _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, _opti.x)
                    
                    _scale = _opti.x[0]
                    _pr = _opti.x[1]
                    
                    _Bcdf = lambda x, *pa: cdf(x, nprGaussians, *pa)
                    KS = kstest(_pdata, _Bcdf, (_max, _max/100, _num, _q, _ws, _scale, _pr))
                    
                    #_resid = _opti.fun
                    #_chiSq = chisquare(hy, hy + _resid)
                    
                    # BI for binomial individual fit
                    _pr_results = [_ROI, _ID, _num, _pr, _scale, _ws, _q, "K-S", KS.statistic, KS.pvalue, "BI" ]
                    self.fitInfo.appendOutText (linePrint(_pr_results, pre=4), "darkmagenta")
                    self.saveBtn.setEnabled(True)
                    
                    # display the fit
                    _c = target.plot(_hx_u, _hy_u, name='{} Individual Binomial fit: {} Gaussians, Pr: {:.3f}'.format(_ROI, _num,_pr,_q))
                    
                    # from pyqtgraph.examples
                    _c.setPen(color=col_series, width=3)
                    _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                    
                    # save results to dataframe
                    self.currentROIFits.loc[imax + 1, (_condition, slice(None))]= _pr_results
                    
                # histogram was made for each set now so we can set the same maximum y for all
                for t in self.hPlot.stackMembers:
                    t.setYRange(0, _ymax*1.2)
                
             
             
            if self.fitHistogramsOption == "Individual":
                # if fits were done, they are complete so show results
                self.outputF.appendOutText ("results:\n {}".format(linePrint(self.currentROIFits.iloc[-1])))
            
            if "Global" in self.fitHistogramsOption:
                # put both global options together and use common code
                _hys = np.vstack(ystack).transpose()
                _hxs = np.vstack(xstack).transpose()
                
                # guesses
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                _ws = self.histo_Max_Spin.value() / 10
                if self.fitHistogramsOption == "Global Binom":
                    _opti = fit_nprGaussians_global (_num, _q, _ws, _hys, _hxs)
                else:
                    _opti = fit_PoissonGaussians_global (_num, _q, _ws, _hys, _hxs)
                
                if _opti.success:
                    self.outputF.appendOutText ("Global _opti.x {}\n .cost = {}".format(linePrint(_opti.x, pre=3), _opti.cost), color="Green")
                    _q = _opti.x[0]
                    _ws = _opti.x[1]
                    _scale = _opti.x[2]
                    _resid = _opti.fun.reshape(-1, len(self.peakResults.keys()))
                    imax = self.currentROIFits.index.max()
                    if np.isnan(imax):
                        imax = 0
                    
                    for i, _condition in enumerate(self.peakResults.keys()):
                        # colours
                        col_series = (i, len(self.peakResults.keys()))
                        _hxr = _hxs[:, i]
                        _hxc = np.mean(np.vstack([_hxr[0:-1], _hxr[1:]]), axis=0)
                        target = self.hPlot.stackMembers[i]
                        #_chiSq = chisquare(_hys[i], _hys[i] + _resid [i])
                        _pdata = self.peakResults[_condition][_ROI]
                        if "Binom" in self.fitHistogramsOption:
                            _pr = _opti.x[i+3]
                            
                            _Bcdf = lambda x, *pa: cdf(x, nprGaussians, *pa)
                            KS = kstest(_pdata, _Bcdf, (_max, _max/100, _num, _q, _ws, _scale, _pr))
                            print ("Binomial fit: {}".format(KS))
                            legend = 'Global Binomial Fit {}: {} Gaussians, Pr: {:.3f}, K.-S. P: {:.3f}'.format(_ROI, _ID, _num, _pr, KS.pvalue)
                            _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, [_scale, _pr])
                            _globalR = [_ROI, _ID, _num, _pr, _scale, _ws, _q, "K-S", KS.statistic, KS.pvalue, "BG"]
                            _fitinfoCol = "darkred"
                        
                        elif "Poisson" in self.fitHistogramsOption:
                            _mu = _opti.x[i+3]
                            _Pcdf = lambda x, *pa: cdf(x, poissonGaussians, *pa)
                            KS = kstest(_pdata, _Pcdf, (_max, _max/100, _num, _q, _ws, _scale, _mu))
                            print ("Poisson fit: {}".format(KS))

                            legend = 'Global Poisson Fit {}: {} Gaussians, mu: {:.3f}, K.-S. P: {:.3f}'.format(_ROI, _ID, _num,_mu, KS.pvalue)
                            _hx_u, _hy_u = PoissonGaussians_display (_hxc, _num, _q, _ws, [_scale, _mu])
                            _globalR = [_ROI, _ID, _num, _mu, _scale, _ws, _q, "K-S", KS.statistic, KS.pvalue, "PG"]
                            _fitinfoCol = "darkcyan"
                        
                        self.fitInfo.appendOutText (linePrint(_globalR, pre=4), _fitinfoCol)
                        _c = target.plot(_hx_u, _hy_u, name=legend)
                        _c.setPen(color=col_series, width=3)
                        _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                        
                        self.currentROIFits.loc[imax + 1, (_condition, slice(None))] = _globalR
                        print (self.currentROIFits)
                        self.Fits_data[_ID].addFData(_condition, _hx_u, _hy_u)
                    
                    self.saveBtn.setEnabled(True) # results so we have something to save
                    
                else :
                    self.outputF.appendOutText ("Global fit failed! reason: {} cost: {}".format(_opti.message, _opti.cost), "Red")
                    
                    del self.Fits_data['_ID']
                    
                self.fitHistogramsOption = "None" # to stop cycling but avoid problems with substrings.
                
        elif _hsum == "Summed":
            
            # unite histogram view
            if self.split_state == True:
                
                # remove stack
                self.hPlot.glw.removeItem(self.hPlot.stack)
                # add single
                self.hPlot.glw.addItem(self.hPlot.h)
                # replace in layout *4-tuple has position and size
                #self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
                # when complete, toggle to bypass next time
                self.split_state = False
                
            sumhy = np.zeros(_nbins)
            for _condition in self.peakResults.keys():
                _pdata = self.peakResults[_condition][_ROI].dropna()
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                sumhy += hy
            
            self.hPlot.h.clear()
            self.hPlot.h.plot(hx, sumhy, name="Summed histogram "+_ROI, stepMode=True, fillLevel=0, fillOutline=True, brush='y')
           
            if self.fitHistogramsOption == "Summed":
                #print ("lens hx, hy", len(hx), len(hy))
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                _ws = self.histo_Max_Spin.value() / 20
               
                _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                _opti = fit_nGaussians(_num, _q, _ws, sumhy, _hxc)
                _hx_u, _hy_u = nGaussians_display (_hxc, _num, _opti.x)
                _qfit = _opti.x[0]
                _wsfit = _opti.x[1]
                _c = self.hPlot.h.plot(_hx_u, _hy_u, name='Fit {} Gaussians q: {:.3f}, w: {:.3f}'.format(_num, _qfit, _wsfit))
                
                # store fitted parameters for use in separated Pr fit
                self.fixq = _qfit
                self.fixws = _wsfit
                
                # from pyqtgraph.examples
                _c.setPen('w', width=3)
                _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
            
                # as long as we have a fit, we can enable the separate fit button
                if _opti.success:
                    self.reFitSeparateBtn.setEnabled(True)
                    self.outputF.appendOutText ("Summed Histogram fit\nopti.x: {}\nCost: {}".format(linePrint(_opti.x), _opti.cost), "darkgreen")
            
        
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test = True
            print ("test mode: auto load data")
            
    app = QApplication([])
    main_window = histogramFitDialog()
    
    if test:
        ### Test code
        tdata = testData()
        tdata.open_file()
        main_window.addData(tdata.histo_df)
        main_window.loadBtn.setDisabled(True)
        main_window.filename = tdata.filename
        main_window.dataname_label.setText("TEST: {}".format(tdata.filename))
    
    else:
        main_window.loadBtn.setEnabled(True)
    
    main_window.show()
    sys.exit(app.exec_())
