import sys
import datetime
import os.path
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QFileDialog, QApplication, QMainWindow, QGridLayout, QGroupBox, QWidget, QPushButton, QLayout, QDialog, QLabel, QRadioButton, QVBoxLayout
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
    # at the moment, if test data is loaded automatically, you can't load any more data
    def __init__(self, *args, **kwargs):
        self.open_file()
        
    def open_file(self):
        self.filename = "BK_ROI_Trial.xlsx"
      
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
        bsize = (75, 25)
        for b in buttonList:
            b.setFixedSize(*bsize)
        
        _storeAdvBtn = QPushButton('Next (Keep fit results)')
        buttonList.append(_storeAdvBtn)
        _storeAdvBtn.setFixedWidth(200)
        
        _skipBtn = QPushButton('Next (Discard any fits)')
        buttonList.append(_skipBtn)
        _skipBtn.setFixedWidth(200)

        posn = [(0,1), (0,2), (0,3), (0,4), (0,0), (1,0)]
        
        for counter, btn in enumerate(buttonList):
            
            btn.pressed.connect(lambda val=counter: self.buttonPressed(val))
            #btn.clicked.connect(self.ctrl_signal.emit)
            #self.ctrl_signal.connect(parent.ctrl_signal.emit)
            l.addWidget(btn, *posn[counter])
          
        self.ROI_label = QtGui.QLabel("-")
        self.ROI_label.setFixedSize(300, 25)
        #self.filename_label = QtGui.QLabel("No file")
        l.addWidget(self.ROI_label, 1, 1, 1, 4)
        #l.addWidget(self.filename_label, 0, 5)
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
        
        self.glw = pg.GraphicsLayoutWidget()
        self.hrc = (0, 0, 1, 1) #same as below
        self.h = self.glw.addPlot(title="<empty>", *self.hrc)
        self.h.setLabel('left', "N")
        self.h.setLabel('bottom', "dF / F")
        self.h.vb.setLimits(xMin=0, yMin=0)
        self.h.addLegend()
        self.stack = pg.GraphicsLayout()
          
    def updateTitle(self, newTitle):
        self.h.setTitle (newTitle)
        
    def createSplitHistLayout(self, keys):
        """separated view with each Histogram stacked in separate plots"""
        # adapted from SAFT
        # Store the plot items in a list - can't seem to get them easily otherwise?
        data = []   # empty
        self.stackMembers = []
        for _set in keys:
        
            memberName = _set + " histogram"
            stack_member = self.stack.addPlot(y=data, name=memberName)
            
            # position the title within the frame of the graph
            title = pg.TextItem(_set)#, offset=(70,30))
            title.setPos(60, 5)
            title.setParentItem(stack_member)
            
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

    def appendOutText(self, newOP=None):
        if newOP != None:
            self.frame.append(str(newOP))

    def size(self, _w=300, _h=600):
        self.frame.resize(_w, _h)
        self.frame.setMinimumSize(_w, _h)
        self.frame.setMaximumSize(_w, _h)
    
class histogramFitDialog(QDialog):
    
    ctrl_signal = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super(histogramFitDialog, self).__init__(*args, **kwargs)
        _now = datetime.datetime.now()
        
        self.fitHistogramsOption = False
        
        self.outputHeader = "{} Logfile\n".format(_now.strftime("%y%m%d-%H%M%S"))
        self.hPlot = HDisplay()
        self.filename = None
        self.outputF = txOutput(self.outputHeader)
        self.makeDialog()
    
    def test(self, sender):
        print (sender)
        self.outputF.appendOutText ('ctrl_button was pressed {}'.format(sender))
    
    def makeDialog(self):
        """Create the controls for the dialog"""
        
        # work through each ROI in turn - fit summed histogram and then convert to quanta from peaks list
        
        self.setWindowTitle("Fit Quantal Histograms")
        self.resize(1000, 800)
        
        # panel for file commands and information
        _fileOptions = QGroupBox("File")
        _fileGrid = QGridLayout()
        
        self.loadBtn = QPushButton('Load')
        self.loadBtn.clicked.connect(self.openData)
        self.saveBtn = QPushButton('Save')
        self.saveBtn.clicked.connect(self.save)
        self.saveBtn.setDisabled(True)
        self.filename_label = QtGui.QLabel("No file")
        
        _fileGrid.addWidget(self.loadBtn, 0, 0, 1, 1)
        _fileGrid.addWidget(self.saveBtn, 1, 0, 1, 1)
        _fileGrid.addWidget(self.filename_label, 0, 1, 2, 1)
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
        self.histo_Max_Spin.valueChanged.connect(self.updateHistograms)
        
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
        self.histo_nG_Spin = pg.SpinBox(value=9, step=1, bounds=[1,20], delay=0, int=True)
        self.histo_nG_Spin.setFixedSize(80, 25)
        self.histo_nG_Spin.valueChanged.connect(self.updateHistograms)
        
        _histw_label = QtGui.QLabel("dF (q) guess")
        _histw_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histo_q_Spin = pg.SpinBox(value=.05, step=0.005, bounds=[0.005,1], delay=0, int=False)
        self.histo_q_Spin.setFixedSize(80, 25)
        self.histo_q_Spin.valueChanged.connect(self.updateHistograms)
        
        _reFit_label = QtGui.QLabel("fixed dF (q), w")
        self.reFitSeparateBtn = QPushButton('Fit separated')
        self.reFitSeparateBtn.clicked.connect(self.reFitSeparated)
        self.reFitSeparateBtn.setDisabled(True)
        
        _sumFit_label = QtGui.QLabel("free dF (q), w")
        _doFitBtn = QPushButton('Fit Summed')
        _doFitBtn.clicked.connect(self.fitGaussians)
        
        _fitGrid.addWidget(_histnG_label, 0, 0)
        _fitGrid.addWidget(self.histo_nG_Spin, 0, 1)
        _fitGrid.addWidget(_histw_label, 1, 0)
        _fitGrid.addWidget(self.histo_q_Spin, 1, 1)
        _fitGrid.addWidget(_doFitBtn, row=2, col=0)
        _fitGrid.addWidget(_sumFit_label, row=2, col=1)
        _fitGrid.addWidget(self.reFitSeparateBtn, row=3, col=0)
        _fitGrid.addWidget(_reFit_label, row=3, col=1)
        
        _fittingPanel.setLayout(_fitGrid)
        
        # histogram view layout
        self.hlayout = QGridLayout()
        
        _fileOptions.setFixedSize(400, 100)
        self.hlayout.addWidget(_fileOptions, 0, 2, 1, 2)
        
        _histOptions.setFixedSize(250, 150)
        self.hlayout.addWidget(_histOptions, 2, 1, 1, 1)
        
        _fittingPanel.setFixedSize(350, 150)
        self.hlayout.addWidget(_fittingPanel, 2, 0, 1, 1)
        
        # histogram view
        self.histogramLayPos = (0, 0, 2, 2)
        self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
        
        # ROI controls
        self.RC = ROI_Controls(self)        #need to send this instance as parent
        self.RC.ROI_box.setFixedSize(600, 80)
        self.hlayout.addWidget(self.RC.ROI_box, 3, 0, 1, 2)
       
        # text output console
        self.outputF.size(400,700)
        self.hlayout.addWidget(self.outputF.frame, 1, 2, -1, 3)
        
        self.setLayout(self.hlayout)
        
        
    
    def histogram_parameters(self):
        _nbins = int(self.histo_NBin_Spin.value())
        _max = self.histo_Max_Spin.value()
        self.outputF.appendOutText ("N_bins {}, Max {}".format(_nbins, _max))
        return _nbins, _max
    
    def skipROI(self):
        self.outputF.appendOutText ("Discard fit results from {}".format(self.current_ROI))
        self.Pr_by_ROI.pop(self.current_ROI, None)
            
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI))
        
    def save(self):
        
        _output = pd.DataFrame(self.Pr_by_ROI)
        
        #maybe we just have a filename not a path
        if os.path.split(self.filename)[0] is not None:
            _outfile = os.path.split(self.filename)[0] + "Pr_" + os.path.split(self.filename)[1]
        else :
            _outfile = "Pr_" + self.filename
            
        print (_outfile)
        _output.to_excel(_outfile)
        self.outputF.appendOutText ("write data out to disk {}".format(_outfile))
    
    def storeAdvance(self):
        """storing data and moving forward???"""
        self.outputF.appendOutText ("Keep results for {} --\n Pr : {}".format(self.current_ROI, self.Pr_by_ROI[self.current_ROI]))
        self.ROI_change_command(2)
        self.outputF.appendOutText ("Advance to next ROI: {}".format(self.current_ROI))
    
    def reFitSeparated(self):
        """obtain Pr using binomial, using q and w from previous fit"""
        
        # use last df_Q from last summed fit
        _q = self.fixq
        self.outputF.appendOutText ("Refit separate histograms ({}) with fixed q {:.3f} and binomial Pr.".format(self.current_ROI, _q))
        
        self.fitHistogramsOption = True
        self.sum_hist.setCurrentIndex(1)    #calls updateHistogram and fit
    
            
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
        # called by the fit summed button
        
        self.fitHistogramsOption = True
        _fitSum =self.sum_hist.currentIndex()
        
        if _fitSum == 0:
            self.updateHistograms()             # no need to adjust view, just update and fit
        else:
            self.sum_hist.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.
        
    def openData(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open Data", os.path.expanduser("~"))[0]
        
        if self.filename:
            #"None" reads all the sheets into a dictionary of data frames
            self.histo_df = pd.read_excel(self.filename, None, index_col=0)
            self.addData (self.histo_df)
            self.outputF.appendOutText ("Opening file {}".format(self.filename))
        
    def addData(self, data):
        """Bring in external data for analysis"""
        
        
        if self.filename:
            #show only the filename not the entire path
            _f = os.path.split(self.filename)[1]
            self.filename_label.setText (_f)
          
        #take all peak lists
        self.peakResults = data # a dict of DataFrames
        self.Pr_by_ROI = {}     # dictionary for Pr values from fitting
        
        # any histograms aren't much use, we will change binning, so remove them
        if 'histograms' in self.peakResults:
            del self.peakResults['histograms']
        
        # clean out any low SNR data _ avoid interating over ordered dict
        for key in self.peakResults.copy():
            if "SNR<" in key:
                del self.peakResults[key]
                self.outputF.appendOutText ("Removed low SNR data {}".format(key))
        
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
        
        
        self.hPlot.createSplitHistLayout(self.peakResults.keys())
        
        self.ROI_change()   #default is the first (0).
        self.updateHistograms()
    
    def ROI_change_command (self, button_command):
        print("Button command: {}".format(button_command))
        
        # turn off separate fitting when moving to new ROI
        self.reFitSeparateBtn.setDisabled(True)
        
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
            self.storeAdvance() # comes back here with b_c = 2
            return
        elif button_command == 5:
            self.skipROI()  # comes back here with b_c = 2
            return
            
        print ("self_ROI_N is ", self.ROI_N)
        self.ROI_change(self.ROI_N)
        
    def ROI_change(self, _ROI=0):
        self.ROI_N = _ROI
        self.current_ROI = self.ROI_list[_ROI]
        self.RC.update_ROI_label("{} : {} of {}".format(self.current_ROI, self.ROI_N + 1, len(self.ROI_list)))
        
        #for any change of ROI, the default view is the sum of all histograms
        _fitSum =self.sum_hist.currentIndex()
        
        if _fitSum == 0:
            self.updateHistograms()             # if the veiw was already "summed", no need to adjust, just update and fit
        else:
            self.sum_hist.setCurrentIndex(0)    # sets view to summed, calls update histograms and performs the fit.
        
    def updateHistograms(self):
        """called when histogram controls are changed"""
           
        # get values from controls and summarise to terminal
        _nbins, _max = self.histogram_parameters()
        _ROI = self.current_ROI
        _hsum = self.sum_hist.currentText()
        self.outputF.appendOutText ("Update {0} Histogram(s) for {1} with Nbins = {2} and maximum dF/F = {3}.".format(_hsum, _ROI, _nbins, _max))
    
        if _hsum == "Separated":
            _num = self.histo_nG_Spin.value()
            _pr_results = [_num]
            
            if self.split_state == False:
                # bring in split histogram view
                # remove single
                self.hPlot.glw.removeItem(self.hPlot.h)
                # add multiple
                self.hPlot.glw.addItem(self.hPlot.stack)
                # replace in layout *4-tuple has position and size
                self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
                # when complete, toggle to bypass next time
                self.split_state = True
            
            for i, _set in enumerate(self.peakResults.keys()):
                # colours
                col_series = (i, len(self.peakResults.keys()))
                # get relevant peaks data for displayed histograms
                _pdata = self.peakResults[_set][_ROI]
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                
                # replot in the right place in the stack
                target = self.hPlot.stackMembers[i]
                target.clear()
                target.plot(hx, hy, name="Histogram "+_set, stepMode=True, fillLevel=0, fillOutline=True, brush=col_series)
            
                if self.fitHistogramsOption:
                    # binomial path
                    _q = self.fixq
                    _num = self.histo_nG_Spin.value()
                    _ws = self.fixws # / 1.5              # arbitrary but always too fat!
                    _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                    _opti = fit_nprGaussians (_num, _q, _ws, hy, _hxc)
                    _hx_u, _hy_u = nprGaussians_display (_hxc, _num, _q, _ws, _opti)
                    _pr = _opti.x[0]
                    _pr_results.append([_set, _pr, _opti.cost])
                    print ('opti[x]: {}'.format(_opti.x))
                    self.saveBtn.setEnabled(True)
                    
                    # display the fit
                    _c = target.plot(_hx_u, _hy_u, name='Fit {} Gaussians, Pr: {:.3f}, q: {:.3f}'.format(_num,_pr,_q))
                    
                    # from pyqtgraph.examples
                    _c.setPen(color=col_series, width=3)
                    _c.setShadowPen(pg.mkPen((70,70,30), width=8, cosmetic=True))
                    self.Pr_by_ROI[self.current_ROI] = _pr_results # continuous updating/overwriting
                    
                self.outputF.appendOutText ("self PR by ROI: {}".format(self.Pr_by_ROI))
        
        elif _hsum == "Summed":
            
            # unite histogram view
            if self.split_state == True:
                
                # remove stack
                self.hPlot.glw.removeItem(self.hPlot.stack)
                # add single
                self.hPlot.glw.addItem(self.hPlot.h)
                # replace in layout *4-tuple has position and size
                self.hlayout.addWidget(self.hPlot.glw, *self.histogramLayPos)
                # when complete, toggle to bypass next time
                self.split_state = False
                
            sumhy = np.zeros(_nbins)
            for _set in self.peakResults.keys():
                _pdata = self.peakResults[_set][_ROI]
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                sumhy += hy
            
            self.hPlot.h.clear()
            self.hPlot.h.plot(hx, sumhy, name="Summed histogram "+_ROI, stepMode=True, fillLevel=0, fillOutline=True, brush='y')
           
            if self.fitHistogramsOption:
                #print ("lens hx, hy", len(hx), len(hy))
                _num = self.histo_nG_Spin.value()
                _q = self.histo_q_Spin.value()
                _ws = self.histo_Max_Spin.value() / 20
               
                _hxc = np.mean(np.vstack([hx[0:-1], hx[1:]]), axis=0)
                _opti = fit_nGaussians(_num, _q, _ws, sumhy, _hxc)
                _hx_u, _hy_u = nGaussians_display (_hxc, _num, _opti)
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
                    self.outputF.appendOutText ("opti.x: {} opti.cost {}".format(_opti.x, _opti.cost))
            
        
if __name__ == '__main__':
    
    app = QApplication([])
    main_window = histogramFitDialog()
    main_window.show()
    
    ### Test specific code
    tdata = testData()
    tdata.open_file()
    main_window.addData(tdata.histo_df)
    main_window.loadBtn.setDisabled(True)
    main_window.filename = tdata.filename
    main_window.filename_label.setText("TEST: {}".format(tdata.filename))
    
    sys.exit(app.exec_())
