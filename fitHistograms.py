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
from utils import getRandomString, linePrint, txOutput
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
        bsize = (80, 20)
        for b in buttonList:
            b.setFixedSize(*bsize)
        
        _clearFitsBtn = QPushButton('Clear recent fits')
        buttonList.append(_clearFitsBtn)
        _clearFitsBtn.setFixedWidth(150)
        
        _storeAdvBtn = QPushButton('Next ROI, keep fits')
        buttonList.append(_storeAdvBtn)
        _storeAdvBtn.setFixedWidth(150)
        
        _skipBtn = QPushButton('Next ROI, discard fits')
        buttonList.append(_skipBtn)
        _skipBtn.setFixedWidth(150)

        posn = [(0,0,1,3), (0,3,1,3), (0,6,1,3), (0,9,1,3), (1,0,1,4), (1,4,1,4), (1,8,1,4)]
        
        for counter, btn in enumerate(buttonList):
            
            btn.pressed.connect(lambda val=counter: self.buttonPressed(val))
           
            #self.ctrl_signal.connect(parent.ctrl_signal.emit)
            l.addWidget(btn, *posn[counter])
          
        self.ROI_label = QtGui.QLabel("-")
        self.ROI_label.setFixedSize(250, 40)
        self.ROI_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        l.addWidget(self.ROI_label, 2, 0, 1, -1)
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


    
class histogramFitDialog(QDialog):
    
    ctrl_signal = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super(histogramFitDialog, self).__init__(*args, **kwargs)
        _now = datetime.datetime.now()
        
        #can be set to "None" following a long fit routine to stop cycling
        self.fitHistogramsOption = "Summed"
        self.outputHeader = "{} Logfile\n".format(_now.strftime("%y%m%d-%H%M%S"))
        self.hPlot = HDisplay()
        self.autoSave = True        # must match initial state of checkbox!
        self.saveFits = True        # must match initial state of checkbox!
        self.filename = None
        self.dataname = None
        self.fitPColumns = ['ROI', 'Fit ID', 'N', 'Pr/mu', 'Events', 'Width', 'dF_Q', 'Test', 'Stat.', 'P_val', 'Type']
        self.fitInfoHeader = linePrint(self.fitPColumns)
        self.outputF = txOutput(self.outputHeader)
        self.current_ROI = None
        self.ROI_SD = 0                 # the SD for the ROI (assumed similar over conditions)
        self.maxFlag = "Auto max"          # how to find histogram x-axis
        self.fixW = False
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
        self.doneBtn = QPushButton('Done')
        self.doneBtn.clicked.connect(self.done)
        self.doneBtn.setDisabled(True)
        
        self.dataname_label = QtGui.QLabel("No data")
        
        self.autoSaveSwitch = QCheckBox('Auto-save fit results')
        self.autoSaveSwitch.setChecked(True)
        self.autoSaveSwitch.stateChanged.connect(self.toggleAutoSave)
        
        self.sHFCurvesSwitch = QCheckBox('Save Histogram Fitted Curves')
        self.sHFCurvesSwitch.setChecked(True)
        self.sHFCurvesSwitch.stateChanged.connect(self.toggleSaveFits)
        
        _fileGrid.addWidget(self.loadBtn, 0, 0, 1, 2)
        _fileGrid.addWidget(self.saveBtn, 0, 2, 1, 2)
        _fileGrid.addWidget(self.doneBtn, 0, 4, 1, 2)
        _fileGrid.addWidget(self.autoSaveSwitch, 2, 0, 1, 3)
        _fileGrid.addWidget(self.dataname_label, 1, 0, 1, -1)
        _fileGrid.addWidget(self.sHFCurvesSwitch, 2, 3, 1, 3)
        _fileOptions.setLayout(_fileGrid)
        
        # panel of display options for histograms
        _histOptions = QGroupBox("Histogram options")
        _histGrid = QGridLayout()
        
        _NBin_label = QtGui.QLabel("No. of bins")
        _NBin_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_NBin_Spin = pg.SpinBox(value=100, step=10, bounds=[10, 250], delay=0)
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
        self.sum_hist_option = pg.ComboBox()
        self.sum_hist_option.setFixedSize(100,25)
        self.sum_hist_option.addItems(['Summed', 'Separated'])
        self.split_state = False #because summed is first in list and default
        self.sum_hist_option.currentIndexChanged.connect(self.updateHistograms)
        
        _histGrid.addWidget(_NBin_label, 2, 0)
        _histGrid.addWidget(_histMax_label, 1, 0)
        _histGrid.addWidget(_histsum_label, 0, 0)
        _histGrid.addWidget(self.histo_NBin_Spin, 2, 1)
        _histGrid.addWidget(self.histo_Max_Spin, 1, 1)
        _histGrid.addWidget(self.sum_hist_option, 0, 1)
        _histOptions.setLayout(_histGrid)
        
        # panel of fit parameters and commands
        _fittingPanel = QGroupBox("Fitting")
        _fitGrid = QGridLayout()

        

        _hist_W_label = QtGui.QLabel("Gaussian widths (from SD)")
        self.histo_W_Spin = pg.SpinBox(value=self.ROI_SD, step=0.005, delay=0, int=False)
        self.histo_W_Spin.setFixedSize(80, 25)
        
        _hist_nG_label = QtGui.QLabel("Gaussian components")
        #_hist_nG_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_nG_Spin = pg.SpinBox(value=6, step=1, bounds=[1,20], delay=0, int=True)
        self.histo_nG_Spin.setFixedSize(50, 25)
        self.histo_nG_Spin.setAlignment(QtCore.Qt.AlignRight)
        #print (self.histo_nG_Spin.alignment())
        #self.histo_nG_Spin.valueChanged.connect(self.updateHistograms)          ###SHOULD IT?
        
        _hist_q_label = QtGui.QLabel("Quantal size (dF)")
        #_hist_q_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_q_Spin = pg.SpinBox(value=.05, step=0.005, bounds=[0.005,1], delay=0, int=False)
        self.histo_q_Spin.setFixedSize(80, 25)
        self.histo_q_Spin.setAlignment(QtCore.Qt.AlignRight)
        #self.histo_q_Spin.valueChanged.connect(self.updateHistograms)
        
        self.fixWtoSDSwitch = QCheckBox('Fix W according to SD')
        self.fixWtoSDSwitch.setChecked(False)
        self.fixWtoSDSwitch.setDisabled(True)
        self.fixWtoSDSwitch.stateChanged.connect(self.toggleFixW)
        
        _separateBinomial_label = QtGui.QLabel("fixed dF (q), w")
        self.separateBinomialBtn = QPushButton('Separate Binomial fits')
        self.separateBinomialBtn.clicked.connect(self.separateBinomialFits)
        self.separateBinomialBtn.setDisabled(True)
        
        _sumFit_label = QtGui.QLabel("free dF (q), w, amplitudes")
        _doFitBtn = QPushButton('Fit Summed')
        _doFitBtn.clicked.connect(self.fitGaussians)
        
        _globFit_label = QtGui.QLabel("find N, Pr; common dF (q), w")
        _globFitBtn = QPushButton('Global Binomial fit')
        _globFitBtn.clicked.connect(self.binomialFitGlobalGaussians)
        
        _PoissonGlobalFit_label = QtGui.QLabel("find mu; common dF (q), w")
        _PoissonGlobalFitBtn = QPushButton('Global Poisson fit')
        _PoissonGlobalFitBtn.clicked.connect(self.poissonFitGlobalGaussians)
        
        _fitInfoLabel = QtGui.QLabel("Fits for current ROI")
        _fitInfoLabel.setFixedSize(120,20)
        
        self.fitInfo = txOutput(self.fitInfoHeader)
        self.fitInfo.size(480, 130)
        
        _fitGrid.addWidget(_hist_nG_label, 1, 0)
        _fitGrid.addWidget(self.histo_nG_Spin, 1, 1)
        _fitGrid.addWidget(_hist_q_label, 2, 0)
        _fitGrid.addWidget(self.histo_q_Spin, 2, 1)
        _fitGrid.addWidget(_hist_W_label, 3, 0)
        _fitGrid.addWidget(self.histo_W_Spin, 3, 1)
        _fitGrid.addWidget(self.fixWtoSDSwitch, 4, 0, 1 ,2)
        
        
        _fitGrid.addWidget(_doFitBtn, 1, 2, 1, 1)
        _fitGrid.addWidget(_sumFit_label, 1, 3, 1, 1)
        _fitGrid.addWidget(self.separateBinomialBtn, 2, 2, 1, 1)
        _fitGrid.addWidget(_separateBinomial_label, 2, 3, 1, 1)
        _fitGrid.addWidget(_globFitBtn, 3, 2, 1, 1)
        _fitGrid.addWidget(_globFit_label, 3, 3, 1, 1)
        _fitGrid.addWidget(_PoissonGlobalFitBtn, 4, 2, 1, 1)
        _fitGrid.addWidget(_PoissonGlobalFit_label, 4, 3, 1, 1)
        
        _fitGrid.addWidget(_fitInfoLabel, 0, 5, 1, 1)
        _fitGrid.addWidget(self.fitInfo.frame, 1, 5, -1, 1)
        _fittingPanel.setLayout(_fitGrid)
        
        # histogram analysis layout
        self.hlayout = QGridLayout()
        _secondcolW = 450
        
        # histogram view
        self.histogramLayPos = (0, 0, 4, 1)
        self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
        
        # ROI controls
        self.RC = ROI_Controls(self)        #need to send instance as parent
        self.RC.ROI_box.setFixedSize(_secondcolW, 120)
        self.hlayout.addWidget(self.RC.ROI_box, 1, 1, 1, 1)
        
        # Display options for the histograms
        _histOptions.setFixedSize(_secondcolW, 120)
        self.hlayout.addWidget(_histOptions, 3, 1, 1, 1)
        
        # File controls
        _fileOptions.setFixedSize(_secondcolW, 120)
        self.hlayout.addWidget(_fileOptions, 0, 1, 1, 1)
        
        # Text output console
        self.outputF.frame.setFixedSize(_secondcolW, 250)
        self.hlayout.addWidget(self.outputF.frame, 2, 1, 1, 1)
        
        # Fitting controls and display
        _fittingPanel.setFixedHeight(200)
        self.hlayout.addWidget(_fittingPanel, 4, 0, 1, -1)
        
        
        self.setLayout(self.hlayout)
    
    def setManualMax(self):
        self.maxFlag = "Manual max"
        self.updateHistograms()
    
    def histogramParameters(self, verbose=False):
        """ read bins from GUI and optionally read or calculate x (F) max"""
        
        _nbins = int(self.histo_NBin_Spin.value())
        _hsumOpt = self.sum_hist_option.currentText()
        
        if self.maxFlag == "Manual max":
            _max = self.histo_Max_Spin.value()
            
            if verbose:
                self.outputF.appendOutText ("N_bins {}, manual Max {}".format(_nbins, _max))
        
            
        elif self.maxFlag == "Auto max":
            
            _max = 0
            _ROI = self.current_ROI
            for _condition in self.peakResults.keys():
                _peaks = self.peakResults[_condition][_ROI]
                if _peaks.max() > _max * 1.2:
                    _max = _peaks.max() * 1.2
            
            if verbose:
                self.outputF.appendOutText ("N_bins {}, auto Max {}".format(_nbins, _max))
           
        return _hsumOpt, _nbins, _max
            
    
    def clearFits(self):
    
        self.outputF.appendOutText ("Discarded fit results from {}".format(self.current_ROI), "red")
        
        # clear current fits and info frame
        self.currentROIFits = pd.DataFrame(columns=self.currentROIFits.columns)
        
        self.fitInfo.reset(self.fitInfoHeader)
    
    def skipROI(self):
        
        self.clearFits()
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI), "magenta")
    
    
    def toggleAutoSave(self):
        if self.autoSaveSwitch.isChecked() == False:
            self.autoSave = False
    
        else:
            self.autoSave = True
        print ("AutoSave is {}.".format(self.autoSave))
        
        
    def toggleSaveFits(self):
        if self.sHFCurvesSwitch.isChecked() == False:
            self.saveFits = False
        else:
            self.saveFits = True
        print ("SaveFitsToggle is {}.".format(self.saveFits))
        
        
    def toggleFixW(self):
        if self.fixWtoSDSwitch.isChecked() == False:
            self.fixW = False
            self.histo_W_Spin.setDisabled(False)
        else:
            self.fixW = True
            self.histo_W_Spin.setDisabled(True)
        print ("FixWToggle is {}.".format(self.fixW))
        
    def done(self, *arg):
        
        #print ("done arg {}".format(arg))  ## recursion limit
        
        try:
            self.accept()   # works if the dialog was called from elsewhere
        except:
            print ("Bye.")
            self.hide()     # works if the dialog was called standalone
            
    def save(self, auto=False):
        #maybe we just have a filename not a path
        
        ## override when autosaving
        if auto:
            _saveFilename = "HFtemp.xlsx"
            
        # Save was requested by the user
        else:
            self.outputF.appendOutText ("Keeping {} fit results for {} --\n".format(len(self.currentROIFits.index),self.current_ROI), "Magenta")
        
            self.fitResults = self.fitResults.append(copy.copy(self.currentROIFits), ignore_index=True)
        
            usr = os.path.expanduser("~")
            #print ("user path : {}".format(usr))
            if self.filename:

                if os.path.split(self.filename)[0] is not None:
                    _outfile =  os.path.join(usr,"HFit_" + os.path.split(self.filename)[1])
                    #print ("_outfile : {}".format(_outfile))
                else:
                    _outfile = "HFit_" + self.filename
                
            _saveFilename = QFileDialog.getSaveFileName(self,
                        "Save Results",  _outfile)[0]
            
            if _saveFilename == None:
                print ("File dialog failed.")
                return
        
        #print ("sfn : {}".format(_saveFilename))
        
        with pd.ExcelWriter(_saveFilename) as writer:
            
            #print (self.fitResults.head(5))
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

        if auto:
            print ("Autosaved to {}".format(_saveFilename))
        else:
            print ("Wrote {} to disk.".format(_outfile))
            self.outputF.appendOutText ("Wrote fit results out to disk: {}".format(_outfile))
       

    
    def storeAdvance(self):
        """storing data and moving forward one ROI"""
        self.outputF.appendOutText ("Keeping {} fit results for {} --\n".format(len(self.currentROIFits.index),self.current_ROI), "Magenta")
        
        self.fitResults = self.fitResults.append(copy.copy(self.currentROIFits), ignore_index=True)
        if self.autoSave:
            self.save(auto=True)
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI), "Magenta")
        # empty the current fits dataframe
        self.currentROIFits = pd.DataFrame(columns=self.currentROIFits.columns)
        self.fitInfo.reset(self.fitInfoHeader)
        
    
    
            
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
        
    def fitUpdateLogic(self):
        """if histograms are displayed summed, they must be switched to the separated view for separate or global fits"""
       
        if self.sum_hist_option.currentIndex() == 1:
            # no need to adjust view (histograms already separate), just update and fit
            self.updateHistograms()
        else:
            # sets view to separate histograms, calls update histograms, which also performs the fit.
            self.sum_hist_option.setCurrentIndex(1)
    
    def separateBinomialFits(self):
        """Target of the separate binomial fit button
        obtain Pr from separated histograms using binomial,
        using q from previous fit,
        w is guessed or fixed to supplied value"""
        # will use last df_Q from last summed fit (done automatically upon changing ROI)
        _q = self.fixq
        self.outputF.appendOutText ("Fit separate histograms ({}) with fixed q {:.3f} and binomial Pr.".format(self.current_ROI, _q))
        self.fitHistogramsOption = "Individual"
        self.fitUpdateLogic() # updates display if necessary and performs the fit.
    
    def poissonFitGlobalGaussians(self):
        """Target of the global poisson fit button
        obtain mean release rate using Poisson, estimating quantal size and scale, option to fix or estimate w"""
        
        self.fitHistogramsOption = "Global Poisson"
        self.outputF.appendOutText ("\nGlobal Poisson Fit", "darkcyan")
        self.fitUpdateLogic() # updates display if necessary and performs the fit.
    
    def binomialFitGlobalGaussians(self):
        """Target of the global binomial fit button
        obtain Pr using binomial, estimating quantal size and scale, option to fix or estimate w"""
        
        self.fitHistogramsOption = "Global Binom"
        self.outputF.appendOutText ("\nGlobal Binomial Fit", "darkred")
        self.fitUpdateLogic() # updates display if necessary and performs the fit.
        
    def fitGaussians(self):
        """Target of the fit summed button"""
        self.fitHistogramsOption = "Summed"
        if self.sum_hist_option.currentIndex() == 0:
            self.updateHistograms()             # no need to adjust view, just update and fit
        else:
            self.sum_hist_option.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.

    def openData(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open Data", os.path.expanduser("~"))[0]
        
        if self.filename:
            #"None" reads all the sheets into a dictionary of data frames
            self.open_df = pd.read_excel(self.filename, None, index_col=0)
            self.addData (self.open_df)
            self.outputF.appendOutText ("Opening file {}".format(self.filename))
        
    def addData(self, _data, _name=None, _SD=None):
        """
        Bring in external data for analysis
        _data should be a dictionary of dataframes
        keys are conditions
        _SD is a dictionary with ROIs as keys, SD as values
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
        
        self.SD = _SD       #None by default
        
        #take all peak lists
        self.peakResults = _data # a dict of DataFrames

        
        # any histograms aren't much use, we will change binning, so remove them
        if 'histograms' in self.peakResults:
            del self.peakResults['histograms']
        
        # clean out any low SNR data _ avoid interating over an ordered dict
        for key in self.peakResults.copy():
            if "SNR<" in key:
                del self.peakResults[key]
                self.outputF.appendOutText ("Removed low SNR data {}".format(key), "Red")
        
        pRK = self.peakResults.keys()
        
        print ("Conditions: {}".format(linePrint(pRK)))
        N_ROI = [len (self.peakResults[d].columns) for d in pRK]
        
        self.outputF.appendOutText ("Added data of type {}:\n".format(type(self.peakResults)), "Blue")
        for _c, _d in zip(pRK, [self.peakResults[d].head(3) for d in pRK]):
            self.outputF.appendOutText ("{}\n{}\n".format(_c, _d), "Blue")
        
        for d in self.peakResults:
            self.peakResults[d].rename(despace, axis='columns', inplace=True)
           
        # minimal approach, consider exhaustive list as in SAFT.py
        self.ROI_list = list(self.peakResults[list(pRK)[0]].keys().unique(level=0))
        print ("ROI list: {}".format(linePrint(self.ROI_list)))
        
        self.fitResults = histogramFitParams(pRK, self.fitPColumns)
        self.currentROIFits = histogramFitParams(pRK, self.fitPColumns)
        self.Fits_data = {}
        
        self.hPlot.createSplitHistLayout(self.peakResults.keys())
        
        self.ROI_change()   # default is the first (0).
        self.updateHistograms()
    
    def ROI_change_command (self, button_command):
        #print("Button command: {}".format(button_command))
        
        # turn off separate fitting when moving to new ROI, and get histogram x-range automatically
        self.separateBinomialBtn.setDisabled(True)
        self.maxFlag = "Auto max"
        
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
            
        #print ("self_ROI_N is ", self.ROI_N)
        self.ROI_change(self.ROI_N)
        
    def ROI_change(self, _ROI=0):
        """ update on ROI change"""
        self.ROI_N = _ROI
        self.current_ROI = self.ROI_list[_ROI]
        ROI_label_text = "{} : {} of {}".format(self.current_ROI, self.ROI_N + 1, len(self.ROI_list))
        
        
        if self.SD is not None:
            self.fixWtoSDSwitch.setEnabled(True)
            self.fixws = self.SD[self.current_ROI]
            ROI_label_text += " , baseline SD: {:.3f}".format(self.fixws)
        
        self.RC.update_ROI_label(ROI_label_text)
        
        #for any change of ROI, the default view is the sum of all histograms
        _fitSum =self.sum_hist_option.currentIndex()
        self.fitHistogramsOption = "Summed"
        
        if _fitSum == 0:
            self.updateHistograms()
            # if the view was already "summed", no need to adjust, just update and fit
        else:
            self.sum_hist_option.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.
        
    def updateHistograms(self):
        """Histogram controls were changed, optionally redo the fits.
        """
        
        # would be None if no data was loaded so escape
        if self.current_ROI == None:
            return
        
        _ROI = self.current_ROI
        
        # get values from controls (with auto override)
        _hsum, _nbins, _max = self.histogramParameters()
        self.outputF.appendOutText ("Update {0} Histogram(s) for {1} with {2} bins and maximum dF/F = {3:0.3f}.".format(_hsum, _ROI, _nbins, _max))
    
        # for global fits, we will collect all the histograms and flatten them later (global fits are still 1-D)
        if "Global" in self.fitHistogramsOption:
            ystack = []
            xstack = []
        
        # _ID is a unique identifier for each fit
        # it is the key for the fits dictionary
        _ID = getRandomString(4)
        
        if self.fitHistogramsOption not in ["None", "Summed"]:
            
            self.Fits_data[_ID] = HFStore(self.current_ROI, self.peakResults.keys())
            #print ("SFHO: {}\nSFDID: {}".format(self.fitHistogramsOption, self.Fits_data[_ID]))
         
         
        # Below we draw the individual histograms.
        # at the moment, the individual fits are done as this progresses.
        # global fits must wait until after histograms are all drawn.
        if _hsum == "Separated":
            _num = self.histo_nG_Spin.value()
            
            # bring in split histogram view
            if self.split_state == False:
                
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
                # not all traces have the same number of peaks so drop padding
                _pdata = self.peakResults[_condition][_ROI].dropna()
                #print ("pdata and dropna\n {}\n {}".format(_pdata, _pdata))
                
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                
                if hy.max() > _ymax:
                    _ymax = hy.max()
                
                # Only store histogram values if we are doing a global fit
                if self.fitHistogramsOption not in ["None", "Summed"]:
                   
                    self.Fits_data[_ID].addHData(_condition, hx, hy)
                
                if "Global" in self.fitHistogramsOption:
                    ystack.append(hy)
                    xstack.append(hx)
                
                # replot in the right place in the stack
                target = self.hPlot.stackMembers[i]
                target.clear()   # this unfortunately cleans out any text - we should instead remove the hist + fit?
                target.plot(hx, hy, name="{} {} Histogram".format(_ROI, _condition), stepMode=True, fillLevel=0, fillOutline=True, pen=col_series, brush=col_series)
            
                if self.fitHistogramsOption == "Individual":
                    # separate binomial fits path
                    _q = self.fixq
                    _num = self.histo_nG_Spin.value()
                    
                    if self.fixW and self.SD is not None:   #fix W is ticked and we were given some SDs
                        _ws = self.SD[_ROI]
                        _wsInfo = "Fixed _ws ({}) from supplied value".format(_ws)
                    
                    else:
                        _ws = self.histo_W_Spin.value()
                        _box = "W"
                        if _ws == 0:
                            _ws = self.histo_Max_Spin.value() / 10
                            _box = "M"
                        _wsInfo = "Fixed _ws ({}) from GUI spinbox {}".format(_ws, _box)
                        
                    print ("Fit {}, ROI: {}, _q = {}, {}".format(_ID, _ROI, _q, _wsInfo))
                        
                    _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                    _opti = fit_nprGaussians (_num, _q, _ws, hy, _hxc)
                    #print (_opti)
                    if _opti.success:
                        _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, _opti.x)
                        #print ("hxu, hyu\n{}{}".format(_hx_u, _hy_u))
                        # save fitted curves
                        self.Fits_data[_ID].addFData(_condition, _hx_u, _hy_u)
                    
                        _scale = _opti.x[0]
                        _pr = _opti.x[1]
                    
                        # _Bcdf is a vector of the biomial cdf values for the observed data
                        # KS is the Kolmogorov Smirnoff test of goodness of fit
                        _Bcdf = lambda x, *pa: cdf(x, nprGaussians, *pa)
                        KS = kstest(_pdata, _Bcdf, (_max, _max/_nbins, _num, _q, _ws, _scale, _pr))
                        
                        # IB for binomial individual fit
                        _IB_results = [_ROI, _ID, _num, _pr, _scale, _ws, _q, "KS", KS.statistic, KS.pvalue, "IB" ]
                        self.fitInfo.appendOutText (linePrint(_IB_results, pre=3), "darkmagenta")
                        self.saveBtn.setEnabled(True)
                        
                        # display the fit
                        _c = target.plot(_hx_u, _hy_u, name='Individual Binomial fit: {} Gaussians, Pr: {:.3f}'.format( _num, _pr, _q))
                    
                        # from pyqtgraph.examples
                        _c.setPen(color=col_series, width=3)
                        _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                        
                        # save results to dataframe
                        self.currentROIFits.loc[imax + 1, (_condition, slice(None))]= _IB_results
                    
                    else:
                        self.outputF.appendOutText ("Individual fit failed! reason: {} cost: {}".format(_opti.message, _opti.cost), "Red")
                        # add null fitted curve for consistency, others may have worked fine.
                        self.Fits_data[_ID].addFData(_condition, pd.Series([]), pd.Series([]))
                        
                # histogram was made for each set now so we can set the same maximum y for all
                for t in self.hPlot.stackMembers:
                    t.setYRange(0, _ymax * 1.2)
                
             
            if self.fitHistogramsOption == "Individual":
                # if fits were done, they are complete so show results
                self.outputF.appendOutText ("Individual Fit Results:\n {}".format(linePrint(self.currentROIFits.iloc[-1])), "darkmagenta")
                self.fitHistogramsOption = "None" # to avoid any cycling
            
            # Histograms are complete, so now do any global fit that was requested
            # all global options are together and use common code
            if "Global" in self.fitHistogramsOption:
                
                _hys = np.vstack(ystack).transpose()
                _hxs = np.vstack(xstack).transpose()
                
                # guesses
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                
                # four possibilities : Binomial or poisson, with widths fixed or free
                if self.fitHistogramsOption == "Global Binom" and self.SD is not None and self.fixW:
                    _ws = self.SD[_ROI]
                    print ("Fixed _ws= {}, ROI: {}".format(_ws, _ROI))
                    _opti = fit_nprGaussians_global (_num, _q, _ws, _hys, _hxs, fixedW=True)
                    _fitType = "GBFW"
                    
                elif self.fitHistogramsOption == "Global Binom":
                    
                    if self.SD is None:
                        _ws = self.histo_Max_Spin.value() / 10
                    else:
                        _ws = self.SD[_ROI]
                        
                    _opti = fit_nprGaussians_global (_num, _q, _ws, _hys, _hxs, fixedW=False)
                    _fitType = "GB"
                    
                elif self.fitHistogramsOption == "Global Poisson" and self.SD is not None and self.fixW:
                    _ws = self.SD[_ROI]
                    print ("Fixed _ws= {}, ROI: {}".format(_ws, _ROI))
                    _opti = fit_PoissonGaussians_global (_num, _q, _ws, _hys, _hxs, fixedW=True)
                    _fitType = "GPFW"
                
                elif self.fitHistogramsOption == "Global Poisson":
                    if self.SD is None:
                        _ws = self.histo_Max_Spin.value() / 10
                    else:
                        _ws = self.SD[_ROI]
                        
                    _opti = fit_PoissonGaussians_global (_num, _q, _ws, _hys, _hxs, fixedW=False)
                    _fitType = "GP"
                
                else:
                    print ("fell through sfho: {}".format(self.fitHistogramsOption))
                
                print ("fit type, Ws: {} {}".format(_fitType, _ws))
                
                #if the fit worked
                if _opti.success:
                    
                    self.outputF.appendOutText ("_opti.x: {}\nCost = {}".format(linePrint(_opti.x, pre=3), _opti.cost), color="Green")
                    
                    _q = _opti.x[0]
                    if self.fixW:
                        _scale = _opti.x[1]
                    else:
                        _ws = _opti.x[1]
                        _scale = _opti.x[2]
    
                    imax = self.currentROIFits.index.max()
                    if np.isnan(imax):
                        imax = 0
                    
                    # handle each histogram in turn
                    for i, _condition in enumerate(self.peakResults.keys()):
                        # colours
                        col_series = (i, len(self.peakResults.keys()))
                        _hxr = _hxs[:, i]
                        _hxc = np.mean(np.vstack([_hxr[0:-1], _hxr[1:]]), axis=0)
                        target = self.hPlot.stackMembers[i]
                        
                        _pdata = self.peakResults[_condition][_ROI].dropna()
                        
                        # do Kolmogorov-Smirnoff test of goodness of fit and get the oversampled fitted curves
                        if "Binom" in self.fitHistogramsOption:
                            if _fitType == "GBFW":
                                _pr = _opti.x[i+2]
                            else:
                                _pr = _opti.x[i+3]
                            _Bcdf = lambda x, *pa: cdf(x, nprGaussians, *pa)
                            KS = kstest(_pdata, _Bcdf, (_max, _max/_nbins, _num, _q, _ws, _scale, _pr))
                            
                            legend = 'Global Binomial Fit {}: {} Gaussians, Pr: {:.3f}, K.-S. P: {:.3f}'.format(_ID, _num, _pr, KS.pvalue)
                            _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, [_scale, _pr])
                            _globalR = [_ROI, _ID, _num, _pr, _scale, _ws, _q, "KS", KS.statistic, KS.pvalue, _fitType]
                            _fitinfoCol = "darkred"
                        
                        elif "Poisson" in self.fitHistogramsOption:
                            if _fitType == "GPFW":
                                _mu = _opti.x[i+2]
                            else:
                                _mu = _opti.x[i+3]
                                
                            _Pcdf = lambda x, *pa: cdf(x, poissonGaussians, *pa)
                            KS = kstest(_pdata, _Pcdf, (_max, _max/_nbins, _num, _q, _ws, _scale, _mu))

                            legend = 'Global Poisson Fit {}: {} Gaussians, mu: {:.3f}, K.-S. P: {:.3f}'.format( _ID, _num, _mu, KS.pvalue)
                            _hx_u, _hy_u = PoissonGaussians_display (_hxc, _num, _q, _ws, [_scale, _mu])
                            _globalR = [_ROI, _ID, _num, _mu, _scale, _ws, _q, "KS", KS.statistic, KS.pvalue, _fitType]
                            _fitinfoCol = "darkcyan"
                        
                        # write the results to the running fits text panel
                        self.fitInfo.appendOutText (linePrint(_globalR, pre=3), _fitinfoCol)
                        
                        _c = target.plot(_hx_u, _hy_u, name=legend)
                        _c.setPen(color=col_series, width=3)
                        _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                        
                        self.currentROIFits.loc[imax + 1, (_condition, slice(None))] = _globalR
                        self.Fits_data[_ID].addFData(_condition, _hx_u, _hy_u)
                
                    # results were obtained so we have something to save
                    self.saveBtn.setEnabled(True)
                    
                else :
                    self.outputF.appendOutText ("Global fit failed! reason: {} cost: {} fit type: {}".format(_opti.message, _opti.cost, _fitType), "Red")
                    
                    del self.Fits_data[_ID]
                    print ("Fit failed: {}, removed histogram fit container with ID : {}".format(_opti.message, _ID))
                    
                self.fitHistogramsOption = "None" # to stop cycling but avoid problems with substrings.
                
        elif _hsum == "Summed":
            
            # unite histogram view
            if self.split_state == True:
                
                # remove stack
                self.hPlot.glw.removeItem(self.hPlot.stack)
                # add single
                self.hPlot.glw.addItem(self.hPlot.h)
                # replace in layout *4-tuple has position and size
                #self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos) ###NOT NEEDED
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
                _ws = self.histo_Max_Spin.value() / 10
               
                _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                _opti = fit_nGaussians(_num, _q, _ws, sumhy, _hxc)
                _hx_u, _hy_u = nGaussians_display (_hxc, _num, _opti.x)
                _qfit = _opti.x[0]
                _wsfit = _opti.x[1]
                _c = self.hPlot.h.plot(_hx_u, _hy_u, name='Fit {} Gaussians q: {:.3f}, w: {:.3f}'.format(_num, _qfit, _wsfit))
                
                # store fitted q for use in a separated Pr fit
                self.fixq = _qfit
                
                # from pyqtgraph.examples
                _c.setPen('w', width=3)
                _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
            
                # as long as we have a fit, we can enable the separate fit button
                if _opti.success:
                    self.separateBinomialBtn.setEnabled(True)
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
        main_window.doneBtn.setDisabled(True)
        main_window.filename = tdata.filename
        main_window.dataname_label.setText("TEST: {}".format(tdata.filename))
    
    else:
        main_window.loadBtn.setEnabled(True)
        
    main_window.show()
    sys.exit(app.exec_())
