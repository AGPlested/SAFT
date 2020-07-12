import sys
import os.path
import platform
import copy

from PySide2 import QtCore, QtGui
from PySide2.QtCore import Slot
from PySide2 import __version__ as pyside_version
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QMessageBox, QFileDialog, QAction, QGroupBox, QHBoxLayout, QRadioButton, QDialog, QVBoxLayout, QCheckBox

import itertools
import numpy as np
import pandas as pd
import scipy.signal as scsig
from scipy import sparse
from scipy.sparse.linalg import spsolve

#SAFT imports
from clicker import clickAlgebra
from peaksDialog import getPeaksDialog
from resultsDF import Results
from histogramDF import HistogramsR
from groupPeaksDialog import groupDialog
from quantal import fit_nGaussians, nGaussians_display

import pyqtgraph as pg

# some functions below that could probably go to a module

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """From SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

Not particularly useful for undersampled data!!

Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
The Savitzky-Golay filter removes high frequency noise from data.
It has the advantage of preserving the original shape and
features of the signal better than other types of filtering
approaches, such as moving averages techniques.
Parameters
----------
y : array_like, shape (N,)
    the values of the time history of the signal.
window_size : int
    the length of the window. Must be an odd integer number.
order : int
    the order of the polynomial used in the filtering.
    Must be less then `window_size` - 1.
deriv: int
    the order of the derivative to compute (default = 0 means only smoothing)
Returns
-------
ys : ndarray, shape (N)
    the smoothed signal (or it's n-th derivative).
Notes
-----
The Savitzky-Golay is a type of low-pass filter, particularly
suited for smoothing noisy data. The main idea behind this
approach is to make for each point a least-square fit with a
polynomial of high order over a odd-sized window centered at
the point.
Examples
--------
t = np.linspace(-4, 4, 500)
y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
ysg = savitzky_golay(y, window_size=31, order=4)
import matplotlib.pyplot as plt
plt.plot(t, y, label='Noisy signal')
plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
plt.plot(t, ysg, 'r', label='Filtered signal')
plt.legend()
plt.show()
References
----------
.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
   Data by Simplified Least Squares Procedures. Analytical
   Chemistry, 1964, 36 (8), pp 1627-1639.
.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
   Cambridge University Press ISBN-13: 9780521880688
    """
    #import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        print (str(msg))
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
def baseline_als(y, lam, p, niter=20, quiet=False):
    #from https://stackoverflow.com/questions/29156532
    #from "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
    """There are two parameters: p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand. We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur. In any case one should vary λ on a grid that is approximately linear for log λ"""
    if not quiet: print('Baseline subtraction with lambda {0:.3f} and p {1:.3f}'.format(lam, p))
    L = len(y)
    D = sparse.diags([1, -2, 1],[0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        WW = sparse.spdiags(w, 0, L, L)
        ZZ = WW + lam * D.dot(D.transpose())
        z = spsolve(ZZ, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baselineIterator(data, lam, p, niter=20):
    """ iterate baseline subtraction over dictionary of dataframes of ROI traces """
    
    #is there a problem with running it twice? No, peaks were being chucked out was the problem
    bdata = {}
    
    for _set, df in data.items():
        print ("Auto baseline for {0} set. lambda: {1:.3f} and p: {2:.3f}".format(_set, lam, p))
        
        maxVal = len (df.columns)
        progMsg = "Auto baseline for {0} traces".format(maxVal)
        with pg.ProgressDialog(progMsg, 0, maxVal) as dlg:
            for col in df:
                dlg += 1
                y = np.asarray(df[col])
                #subtract appropriate baseline from each column of df
                df[col] -= baseline_als(y, lam, p, niter=20, quiet=True)
        bdata[_set] = df
    return bdata

def findCurve(items):
    # assume there is one PG PlotDataItem with curve data and return it
    # the others should be empty
    PDIs = [d for d in items if isinstance(d, pg.PlotDataItem)]

    # there should be two plot data items, find the curve data
    for pdi in PDIs:
        x, _ = pdi.curve.getData()
        if len(x) > 0:
            return pdi.curve

def findScatter(items):
    # assume there is one PG PlotDataItem with scatter data and return it
    # the other scatter attributes should be empty
    PDIs = [d for d in items if isinstance(d, pg.PlotDataItem)]
    
    # there should be two plot data items, find the scatter data
    for pdi in PDIs:
        x, _ = pdi.scatter.getData()
        if len(x) > 0:
            return pdi.scatter

def remove_all_scatter(p1):
    """p1 should be a plot item"""
    
    PDIs = [d for d in p1.items if isinstance(d, pg.PlotDataItem)]
    for pdi in PDIs:
        
        x, _ = pdi.scatter.getData()            #scatter data objects have some data points in them
        if len(x) > 0:
            print ("Removing: {}".format(pdi))
            p1.removeItem(pdi)                  #need to use remove item to get rid of it.
            
    _rem = p1.listDataItems()

    print("Data items remaining in {0}: {1}".format(p1, len(_rem)))
            
            
class MainWindow(QMainWindow):
    
    ### Methods
    ###
    ### createMenu                  : make menubar
    ### about                       : About the app
    ### getStarted                  : a help file for novices
    ### createSplitTraceLayout      : for display of split traces
    ### createPlotWidgets           : build the plots
    ### createLinearRegion          : linear selection for zoom window
    ### mouseMoved                  : when the mouse moves in zoom
    ### splitState                  : change the split state for the overview window
    ### manualPeakToggle            :
    ###
    ###
    ###
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle("Semi-Auto Fluorescence Trace analysis")
        self.central_widget = QWidget()
        self.central_layout = QGridLayout()
        self.central_widget.setLayout(self.central_layout)
        self.setCentralWidget(self.central_widget)
        self.resize(1500,800)           # works well on MacBook Retina display
        
        self.df = {}                    # dictionary for trace data frames
        self.extPa = {}                 # external parameters for the peak scraping dialog
        self.LR_created = False         # was a pg linear region created yet?
        self.filename = None
        self.ROI_list = None
        self.dataLoaded = False
        self.simplePeaks = False            # choice of peak finding algorithm
        self.autoPeaks = True                  # find peaks automatically or manually
        self.cwt_width = 5                # width of the continuous wavelet transform peak finding
        self.sheets = ['0.5 mM', '2 mM', '4 mM', '8 mM'] # example with one too many
        self.split_traces = False
        self.dataLock = True                    # when manual peak editing, lock to trace data
        self.noPeaks = True
        
        # setup main window widgets and menus
        self.createPlotWidgets()
        self.createControlsWidgets()
        self.createMenu()
        
        
    def createMenu(self):
        # Skeleton menu commands
        self.file_menu = self.menuBar().addMenu("File")
        self.analysis_menu = self.menuBar().addMenu("Analysis")
        self.help_menu = self.menuBar().addMenu("Help")
        
        self.file_menu.addAction("About FSTPA", self.about) #this actually goes straight into the FSTPA menu
        self.file_menu.addAction("Open File", self.open_file)
        self.file_menu.addAction("Save Peaks", self.save_peaks)
        
        self.file_menu.addAction("Save baselined", self.save_baselined)
        
        self.analysis_menu.addAction("Extract all peaks", self.getResponses)
        self.analysis_menu.addAction("Grouped peak stats", self.getGroups)
        
        self.help_menu.addAction("Getting Started", self.getStarted)
    
    
    def about(self):
        QMessageBox.about (self, "About SAFT",
        """ ----*- SAFT {0} -*----
        \nSemi-Automatic Fluorescence Trace analysis
        \nAndrew Plested 2020
        \nThis application can analyse sets of fluorescence time series.
        \nIt makes heavy use of PyQtGraph (Luke Campagnola).
        \nPython {1}
        \nPySide2 {2} built on Qt {3}
        \nRunning on {4}
        """.format(__version__, platform.python_version(), pyside_version, QtCore.__version__, platform.platform()))
    
    
    def getStarted(self):
        
        helpful_msg = """
        \nWelcome to SAFT - Semi-Automatic Fluorescence Trace analysis
        \n----------------------------
        \nHere is some advice for novice users
        \nLoad your data (traces in columns) from Excel. The first row should name the regions of interest (ROI), and each condition should be a separate sheet: 0.5 mM, 2 mM, 4 mM, etc. The ROIs do not have to be the same for each condition.
        \nAdjust the baseline and find peaks automatically.
        \nIn the 'Peak editing' window, turn on "Edit peaks with mouse". You can manually add peaks by left-clicking (and left-click on existing peaks to remove). Histogram should update as you go. Your clicks are locked to the data. You can do this manually for every trace if you like.
        \nBetter: the "Extract peaks for all ROIs" button will open a dialog that uses the positions of the peaks from the 4 mM "mean" trace to get all the peaks from every other ROI. You can optionally blacklist ROIs from analysis that have a bad SNR. You can also select a region around each peak for the search.
        \nSave the peaks and also the automatically baselined traces from the File menu or buttons. Peaks are sorted in SNR order.
        \nHistograms (summed for each ROI over conditions or separated) can be saved as well.
        \nIf traces consist of grouped responses to repeated stimuli, these can be batch analysed with "Extract group stats".
        """
        QMessageBox.information(self, "Getting Started", helpful_msg)
    
    
    def createSplitTraceLayout(self):
        """Optional split view with each ROI trace in a separate plot (not the default)"""
        
        # Store the plot items in a list - can't seem to get them easily otherwise?
        data = []
        self.p1stackMembers = []
        for c in self.sheets:
            memberName = c + " trace"
            p1_stack_member = self.p1stack.addPlot(title=c, y=data, name=memberName)
            p1_stack_member.hideAxis('bottom')
            self.p1stackMembers.append(p1_stack_member)
            self.p1stack.nextRow()
            #print (c, len(self.p1stackMembers))
        
        #link y-axes - using final member of stack as anchor
        for s in self.p1stackMembers:
            if s != p1_stack_member:
                s.setXLink(p1_stack_member)
                s.setYLink(p1_stack_member)
                
        #add back bottom axis to the last graph
        p1_stack_member.showAxis("bottom")
        p1_stack_member.setLabel('bottom', "Time (s)")
    
    
    def createPlotWidgets(self):
        """analysis plots"""
        
        # traces plot
        data = []
        self.plots = pg.GraphicsLayoutWidget(title="display")
        self.p1rc = (1,0)
        self.p1 = self.plots.addPlot(title="Traces and background subtraction", y=data, row=self.p1rc[0], col=self.p1rc[1], rowspan=3, colspan=1)
        self.p1.setLabel('left', "dF / F")
        self.p1.setLabel('bottom', "Time (s)")
        self.p1.vb.setLimits(xMin=0)
        #just a blank for now, populate after loading data to get the right number of split graphs
        self.p1stack = pg.GraphicsLayout()
        
        if self.dataLoaded:
            createLinearRegion()
        
        # Histograms
        self.p2 = self.plots.addPlot(title="Peak Histograms", row=0, col=0, rowspan=1, colspan=1)
        self.p2.setLabel('left', "N")
        self.p2.setLabel('bottom', "dF / F")
        self.p2.vb.setLimits(xMin=0, yMin=0)
        self.p2.addLegend()
        
        # zoomed editing region
        self.p3 = self.plots.addPlot(title="Peak editing", y=data, row=0, col=1, rowspan=4, colspan=2)
        self.p3.setLabel('left', "dF / F")
        self.p3.setLabel('bottom', "Time (s)")
        self.p3vb = self.p3.vb
        
        # draw the crosshair if we are in manual editing mode
        self.p3proxyM = pg.SignalProxy(self.p3.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
        # what does this do??
        self.p3.scene().sigMouseClicked.connect(self.clickRelay)
        #self.p3.sigMouseClicked.connect(self.clickRelay)
        
        self.plots.cursorlabel = pg.LabelItem(text='', justify='right')
        
        #to fix label jiggling about (graphicswidget method)
        self.plots.cursorlabel.setFixedWidth(100)
        
        self.plots.peakslabel = pg.LabelItem(text='', justify='left')
        
        #to fix label jiggling about (graphicswidget method)
        self.plots.peakslabel.setFixedWidth(100)
        
        self.plots.addItem(self.plots.cursorlabel, row=4, col=2, rowspan=1, colspan=1)
        self.plots.addItem(self.plots.peakslabel, row=4, col=0, rowspan=1, colspan=1)
        
        self.central_layout.addWidget(self.plots, row=0, col=0, rowspan=1,colspan=2)
     
     
    def clickRelay(self, *args):
        """logic to avoid the click signal getting sent out if manual peak editing is not on."""
        if self.autoPeaks:
            print ("Turn on manual peak editing to get some value for your clicks.\nFor debugging: ", args)
            return
        else:
            # the asterisk in call unpacks the tuple into arguments.
            # a free click should be locked to the data (as the crosshair is)
            self.cA.onClick (*args, dataLocked=self.dataLock)
    
    
    def createLinearRegion(self):
        """Linear region in p1 that defines the x-region in p3 (manual editing window)"""
        # taken from pyqtgraph examples.
        
        if self.LR_created == False:
            self.LR_created = True
            xrange = self.ranges['xmax']-self.ranges['xmin']
            self.lr = pg.LinearRegionItem([xrange/2, xrange/1.5])
            self.lr.setZValue(-10)
    
        def updatePlot():
            self.p3.setXRange(*self.lr.getRegion(), padding=0)
        def updateRegion():
            self.lr.setRegion(self.p3.getViewBox().viewRange()[0])
        
        self.lr.sigRegionChanged.connect(updatePlot)
        self.p3.sigXRangeChanged.connect(updateRegion)
        
        if self.split_traces:
            for s in self.p1stackMembers:
                s.addItem(self.lr)
        else:
            self.p1.addItem(self.lr)
        updatePlot()
    
    
    def mouseMoved(self, evt):
        """Crosshair in p3 shown during manual fitting"""
        if self.autoPeaks == False:
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.p3.sceneBoundingRect().contains(pos):
                mousePoint = self.p3vb.mapSceneToView(pos)
                
                # there should be two plot data items, find the curve data
                _c = findCurve(self.p3.items)
                sx, sy = _c.getData()
            
                # quantize x to curve, and get corresponding y that is locked to curve
                idx = np.abs(sx - mousePoint.x()).argmin()
                ch_x = sx[idx]
                ch_y = sy[idx]
                self.hLine.setPos(ch_y)
                self.vLine.setPos(ch_x)
                
                # print ("update label: x={:.2f}, y={:.2f}".format(ch_x, ch_y))
                self.plots.cursorlabel.setText("Cursor: x={: .2f}, y={: .3f}".format(ch_x, ch_y))
    
    
    def splitState(self, b):
        """Called when trace display selection radio buttons are activated """
        if b.text() == "Split traces":
            if b.isChecked() == True:
                self.split_traces = True
            else:
                self.split_traces = False
            
        if b.text() == "Combined traces":
            if b.isChecked() == True:
                self.split_traces = False
            else:
                self.split_traces = True
        
        tobeRemoved = self.plots.getItem(*self.p1rc)
        print ("Removing", tobeRemoved)
        
        #self.p1rc is a tuple containing the position (row-column) of p1
        if self.split_traces:
            self.plots.removeItem(tobeRemoved)
            self.plots.addItem(self.p1stack, *self.p1rc, 3, 1)
        else:
            self.plots.removeItem(tobeRemoved)
            self.plots.addItem(self.p1, *self.p1rc, 3, 1)

        # call general update method
        self.ROI_Change()
    
    
    def saveHistogramsLogic (self, b):
        if b.isChecked() == True:
            self.saveHistogramsOption = True
        else:
            self.saveHistogramsOption = False
    
    def fitHistogramsLogic (self, b):
        if b.isChecked() == True:
            self.fitHistogramsOption = True
        else:
            self.fitHistogramsOption = False
    
    
    def manualPeakToggle (self, b):
        """disable controls if we are editing peaks manually"""
        
        if b.isChecked() == True:
            # enter manual mode
            print ("Manual peak editing")
            self.autoPeaks = False
            
            # disable all controls that could trigger auto peak finding
            self.peak_CB.setDisabled(True)
            self.SGsmoothing_CB.setDisabled(True)
            self.cwt_SNR_Spin.setDisabled(True)
            self.cwt_w_Spin.setDisabled(True)
            self.auto_bs_lam_slider.setDisabled(True)
            self.auto_bs_P_slider.setDisabled(True)
            self.autobs_Box.setDisabled(True)
            self.removeSml_Spin.setDisabled(True)
            
            # Turn on crosshair and change mouse mode in p3.
            # cross hair
            self.vLine = pg.InfiniteLine(angle=90, movable=False)
            self.hLine = pg.InfiniteLine(angle=0, movable=False)
            self.p3.addItem(self.vLine, ignoreBounds=True)
            self.p3.addItem(self.hLine, ignoreBounds=True)
            
            # add a hint - need to find a better way to add it!!! Label at top?
            #self.p3hint = pg.TextItem('Manual editing- left click to add/remove peaks')
            #self.p3hint.setPos(0, .2)
            self.p3.setTitle('Manual editing- left click to add/remove peaks', color="F0F0F0" , width=300)
            
        else:
            # Enter auto peak mode
            print ("Auto peak finding")
            self.autoPeaks = True
            
            # Re-enable all the controls for auto peak finding
            self.peak_CB.setEnabled(True)
            self.SGsmoothing_CB.setEnabled(True)
            self.cwt_SNR_Spin.setEnabled(True)
            self.cwt_w_Spin.setEnabled(True)
            self.auto_bs_lam_slider.setEnabled(True)
            self.auto_bs_P_slider.setEnabled(True)
            self.autobs_Box.setEnabled(True)
            self.removeSml_Spin.setEnabled(True)
            
            # Remove the hint
            self.p3.setTitle('Auto peak finding', color="a0a0a0", width=300)
            
            # Remove crosshair from p3.
            self.p3.removeItem(self.vLine)
            self.p3.removeItem(self.hLine)
     
     
    def createControlsWidgets(self):
        """control panel"""
        
        controls = pg.LayoutWidget()
        
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
        
        #toggle fitting
        self.fitHistogramsToggle = QCheckBox("Fit Histograms", self)
        self.fitHistogramsToggle.setChecked(False)
        self.fitHistogramsToggle.toggled.connect(lambda:self.fitHistogramsLogic(self.fitHistogramsToggle))
        
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
        
        self.saveHistogramsToggle = QCheckBox("Save Histograms", self)
        self.saveHistogramsToggle.setChecked(True)
        self.saveHistogramsToggle.toggled.connect(lambda:self.saveHistogramsLogic(self.saveHistogramsToggle))
        
        histGrid.addWidget(NBin_label, 2, 0)
        histGrid.addWidget(histMax_label, 1, 0)
        histGrid.addWidget(histsum_label, 0, 0)
        histGrid.addWidget(histnG_label, 4, 0)
        histGrid.addWidget(histw_label, 5, 0)
        histGrid.addWidget(self.histo_NBin_Spin, 2, 1)
        histGrid.addWidget(self.histo_Max_Spin, 1, 1)
        histGrid.addWidget(self.sum_hist, 0, 1)
        histGrid.addWidget(self.histo_nG_Spin, 4, 1)
        histGrid.addWidget(self.histo_q_Spin, 5, 1)
        histGrid.addWidget(self.saveHistogramsToggle, 6, 0, 1, 2)
        histGrid.addWidget(self.fitHistogramsToggle, 3, 0, 1, 2)
        histograms.setLayout(histGrid)
        
        traces = QGroupBox("Trace display")
        traceGrid = QGridLayout()
        self.split_B = QRadioButton("Split traces", self)
        self.combine_B = QRadioButton("Combined traces", self)
        self.combine_B.setChecked(True)
        self.split_B.toggled.connect(lambda:self.splitState(self.split_B))
        self.combine_B.toggled.connect(lambda:self.splitState(self.combine_B))
        traceGrid.addWidget(self.combine_B, 0, 0)
        traceGrid.addWidget(self.split_B, 1, 0)
        
        # selection of ROI trace, or mean, variance etc
        ROIBox_label = QtGui.QLabel("Select ROI")
        ROIBox_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        self.ROI_selectBox = QtGui.QComboBox()
        self.ROI_selectBox.addItems(['None'])
        self.ROI_selectBox.currentIndexChanged.connect(self.ROI_Change)
        traceGrid.addWidget(ROIBox_label, 0, 2, -1, 1)
        traceGrid.addWidget(self.ROI_selectBox, 0, 3, -1, 1)
        traces.setLayout(traceGrid)
        
        # peak finding controls box
        peakFinding = QGroupBox("Peak finding and editing")
        pkF_grid = QGridLayout()
        
        # Switch for manual peak finding
        self.manual = QRadioButton("Edit peaks with mouse", self)
        if self.autoPeaks:
            self.manual.setChecked(False)
        self.manual.toggled.connect(lambda:self.manualPeakToggle(self.manual))
        
        p3_select_label = QtGui.QLabel("Show in Peak editing/zoom")
        self.p3Selection = QtGui.QComboBox()
        self.p3Selection.setFixedSize(90, 25)       # only width seems to work
        self.p3Selection.addItems(['-'])
        self.p3Selection.currentIndexChanged.connect(self.ROI_Change)
                
        # Toggle between wavelet transform and simple algorithm for peak finding
        peakFind_L_label = QtGui.QLabel("Find peaks with")
        peakFind_L_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        peakFind_R_label = QtGui.QLabel("algorithm.")
        peakFind_R_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cwt_width_label = QtGui.QLabel("Width (wavelet only)")
        cwt_width_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        SNR_label = QtGui.QLabel("Prominence / SNR")
        SNR_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        self.peak_CB = pg.ComboBox()
        self.peak_CB.setFixedSize(90, 25)
        self.peak_CB.addItems(['wavelet','simple'])
        self.peak_CB.currentIndexChanged.connect(self.ROI_Change)
        
        # spin boxes for CWT algorithm parameters
        self.cwt_SNR_Spin = pg.SpinBox(value=1.5, step=.1, bounds=[.1, 4], delay=0, int=False)
        self.cwt_SNR_Spin.setFixedSize(70, 25)
        self.cwt_SNR_Spin.valueChanged.connect(self.ROI_Change)
        
        self.cwt_w_Spin = pg.SpinBox(value=6, step=1, bounds=[2, 20], delay=0, int=True)
        self.cwt_w_Spin.setFixedSize(70, 25)
        self.cwt_w_Spin.valueChanged.connect(self.ROI_Change)
        
        # Control to exclude small peaks
        removeSml_L_label = QtGui.QLabel("Ignore peaks smaller than")
        removeSml_L_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        removeSml_R_label = QtGui.QLabel("of largest.")
        removeSml_R_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.removeSml_Spin = pg.SpinBox(value=30, step=10, bounds=[0, 100], suffix='%', delay=0, int=False)
        self.removeSml_Spin.setFixedSize(70, 25)
        self.removeSml_Spin.valueChanged.connect(self.ROI_Change)
        
        pkF_grid.addWidget(self.manual, 0, 0)
        pkF_grid.addWidget(p3_select_label, 1, 0)
        pkF_grid.addWidget(self.p3Selection, 1 , 1)
        pkF_grid.addWidget(peakFind_L_label, 2, 0)
        pkF_grid.addWidget(self.peak_CB, 2, 1)
        pkF_grid.addWidget(peakFind_R_label, 2, 2)
        
        pkF_grid.addWidget(cwt_width_label, 4, 0)
        pkF_grid.addWidget(self.cwt_w_Spin, 4, 1)
        pkF_grid.addWidget(SNR_label, 3, 0)
        pkF_grid.addWidget(self.cwt_SNR_Spin, 3, 1)
        
        pkF_grid.addWidget(removeSml_L_label, 5, 0)
        pkF_grid.addWidget(removeSml_R_label, 5, 2)
        pkF_grid.addWidget(self.removeSml_Spin, 5, 1)
        pkF_grid.setSpacing(10)
        pkF_grid.setColumnStretch(0,3)
        pkF_grid.setColumnStretch(2,2)
        peakFinding.setLayout(pkF_grid)
        
        # Baseline controls box
        baseline = QGroupBox("Automatic baseline cleanup")
        base_grid = QGridLayout()
        auto_bs_label = QtGui.QLabel("Baseline removal?")
        self.autobs_Box = pg.ComboBox()
        self.autobs_Box.addItems(['Auto', 'None', 'Lock'])
        self.autobs_Box.setFixedSize(70, 25)
        self.autobs_Box.currentIndexChanged.connect(self.ROI_Change)
        
        # parameters for the auto baseline algorithm
        auto_bs_lam_label = QtGui.QLabel("lambda")
        self.auto_bs_lam_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.auto_bs_lam_slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.auto_bs_lam_slider.setMinimum(2)
        self.auto_bs_lam_slider.setMaximum(9)
        self.auto_bs_lam_slider.setValue(6)
        self.auto_bs_lam_slider.setFixedSize(100, 25)
        self.auto_bs_lam_slider.valueChanged.connect(self.ROI_Change)
        
        auto_bs_P_label = QtGui.QLabel("p")
        self.auto_bs_P_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.auto_bs_P_slider.setMinimum(0)
        self.auto_bs_P_slider.setMaximum(20)
        self.auto_bs_P_slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.auto_bs_P_slider.setValue(3)
        self.auto_bs_P_slider.setFixedSize(100, 25)
        self.auto_bs_P_slider.valueChanged.connect(self.ROI_Change)
        
        # Savitsky-Golay smoothing is very aggressive and doesn't work well in this case
        SGsmoothing_label = QtGui.QLabel("Savitzky-Golay smoothing")
        SGsmoothing_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        #SGsmoothing_label.setFixedSize(170,25)
        self.SGsmoothing_CB = pg.ComboBox()
        self.SGsmoothing_CB.setFixedSize(70, 25)
        self.SGsmoothing_CB.addItems(['Off','On'])
        self.SGsmoothing_CB.currentIndexChanged.connect(self.ROI_Change)
        
        SG_window_label = QtGui.QLabel("Window")
        SG_window_label.setFixedSize(80,25)
        self.SGWin_Spin = pg.SpinBox(value=15, step=2, bounds=[5, 49], delay=0, int=True)
        self.SGWin_Spin.setFixedSize(70, 25)
        self.SGWin_Spin.valueChanged.connect(self.ROI_Change)
        
        
        base_grid.addWidget(auto_bs_label, 0, 0)
        base_grid.addWidget(self.autobs_Box, 0, 1, 1, 3)
        base_grid.addWidget(auto_bs_P_label, 1, 0)
        base_grid.addWidget(self.auto_bs_P_slider, 1, 1, 1, 3)
        base_grid.addWidget(auto_bs_lam_label, 2, 0)
        base_grid.addWidget(self.auto_bs_lam_slider, 2, 1, 1, 3)
        base_grid.addWidget(SGsmoothing_label, 3, 0)
        base_grid.addWidget(self.SGsmoothing_CB, 3, 1)
        base_grid.addWidget(SG_window_label, 3, 2)
        base_grid.addWidget(self.SGWin_Spin, 3, 3)
        base_grid.setColumnStretch(0,1)
        base_grid.setColumnStretch(1,1)
        baseline.setLayout(base_grid)
        

        
        # launch peak extraction wizard dialog
        getResponsesBtn = QtGui.QPushButton('Extract peaks from all ROIs')
        getResponsesBtn.clicked.connect(self.getResponses)
        
        # should be inactive until extraction
        self.savePSRBtn = QtGui.QPushButton('Save peak data')
        self.savePSRBtn.clicked.connect(self.save_peaks)
        self.savePSRBtn.setDisabled(True)
        
        # should be inactive until extraction
        self.save_baselined_ROIs_Btn = QtGui.QPushButton('Save baselined ROI data')
        self.save_baselined_ROIs_Btn.clicked.connect(self.save_baselined)
        self.save_baselined_ROIs_Btn.setDisabled(True)
        
        # should be inactive until extraction
        self.extractGroupsDialog_Btn = QtGui.QPushButton('Average group responses')
        self.extractGroupsDialog_Btn.clicked.connect(self.getGroups)
        self.extractGroupsDialog_Btn.setDisabled(True)
        
        
        dataBtn = QtGui.QPushButton('Show current peak data')
        dataBtn.clicked.connect(self.resultsPopUp)
        
        #stack widgets into control panel
        controls.addWidget(traces, 0, 0, 1, -1)
        controls.addWidget(histograms, 6 , 0, 1, -1)
        
        controls.addWidget(baseline, 1, 0 , 1, -1)
        controls.addWidget(peakFinding, 4, 0 , 2, -1)
        
        controls.addWidget(getResponsesBtn, 7, 0, 1, 2)
        controls.addWidget(self.savePSRBtn, 7, 2, 1, 2)
        
        controls.addWidget(self.save_baselined_ROIs_Btn, 8, 0, 1, 2)
        controls.addWidget(dataBtn, 8, 2, 1, 2)
        controls.addWidget(self.extractGroupsDialog_Btn, 9, 0, 1, 2)
        
        self.central_layout.addWidget(controls, 0, 3, -1, 1)
        return
     
    def resultsPopUp(self):
        """Make a pop up window of the current peak results"""
        _ROI = self.ROI_selectBox.currentText()
        _r = self.peakResults.df[_ROI]
        #print (_r, type(_r))
        qmb = QDialog()
        qmb.setWindowTitle('Peaks from {}'.format(_ROI))
        qmb.setGeometry(1000,600,600,800)
        self.peaksText = QtGui.QTextEdit()
        font = QtGui.QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(12)
        self.peaksText.setCurrentFont(font)
        self.peaksText.setText(_r.to_string())
        self.peaksText.setReadOnly(True)
        
        #add buttons, make it the right size
        qmb.layout = QVBoxLayout()
        qmb.layout.addWidget(self.peaksText)
        qmb.setLayout(qmb.layout)
        qmb.exec_()
    
    def histogram_parameters(self):
        _nbins = int(self.histo_NBin_Spin.value())
        _max = self.histo_Max_Spin.value()
        return _nbins, _max
    
       
    def doHistograms(self):
        """called for histogram output"""
        _nbins, _max = self.histogram_parameters()
        _setList = self.sheets + ["Sum"]
        
        # create a dataframe to put the results in
        self.hDF = HistogramsR(self.ROI_list, _setList, _nbins, 0., _max)
        
        maxVal = len (self.ROI_list) * len (_setList)
        progMsg = "Histogram for {0} traces".format(maxVal)
        with pg.ProgressDialog(progMsg, 0, maxVal) as dlg:
        
            for _set in self.gpd.pkextracted_by_set.keys():      #from the whitelist, should be from edited internal data?
                _pe = self.gpd.pkextracted_by_set[_set]
                print (_set, _pe.columns)
                for _ROI in _pe.columns:
                    dlg += 1
                    # calculate individual histograms and add to dataframe
                    hy, hx = np.histogram(_pe[_ROI], bins=_nbins, range=(0., _max))
                    self.hDF.addHist(_ROI, _set, hy)
                    
        # add sum columns
        self.hDF.ROI_sum()
    
    def updateHistograms(self):
        """called when histogram controls are changed"""
        
        # get controls values and summarise to terminal
        _nbins, _max = self.histogram_parameters()
        _ROI = self.ROI_selectBox.currentText()
        _hsum = self.sum_hist.currentText()
        print ('Update {3} Histogram(s) for {2} with Nbins = {0} and maximum dF/F = {1}.'.format(_nbins, _max, _ROI, _hsum))
       
        # clear
        self.p2.clear()
        
        if _hsum == "Separated":
            for i, _set in enumerate(self.sheets):
                # colours
                col_series = (i, len(self.sheets))
                # get relevant peaks data for displayed histograms
                _, _pdata = self.peakResults.getPeaks(_ROI, _set)
                # redo histogram
                hy, hx  = np.histogram(_pdata, bins=_nbins, range=(0., _max))
                # replot
                self.p2.plot(hx, hy, name="Histogram "+_set, stepMode=True, fillLevel=0, fillOutline=True, brush=col_series)
        
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
        
    def getGroups(self):
        """launch group processing dialog"""
        print ('get group responses from all ROIs.')
        self.getgroupsDialog = groupDialog()
        self.getgroupsDialog.addData(self.gpd.pkextracted_by_set)
        accepted = self.getgroupsDialog.exec_()
        
    def getResponses(self):
        """Wrapping function to get peak data from the dialog"""
        print ('Opening dialog for getting peaks from all ROIs.')
        # if the QDialog object is instantiated in __init__, it persists in state....
        # do it here to get a fresh one each time.
        self.gpd = getPeaksDialog()
        
        # pass the data into the get peaks dialog object
        # we do not want the original trace data modified
        _data = copy.deepcopy(self.df)
        
        # automatically reduce baseline (could also do this interactively??)
        # baselineIterator includes a progress indicator.
        
        # can be Auto or Lock (meaning GUI controls are not updating algorithm)
        if self.autobs_Box.value() != 'None':
            
            if self.autobs_Box.value() == 'Auto':
                # populate values for automatic baseline removal from GUI (unless 'Lock')
                self.setBaselineParams()
            
            _data = baselineIterator(_data, self.auto_bs_lam, self.auto_bs_P)
        
        self.gpd.addData(_data)
        
        #get the times of the peaks that were selected auto or manually
        _peak_t, _ = self.peakResults.getPeaks('Mean', '4 mM')
        print (_peak_t, type(_peak_t))      # pd.series
        _sorted_peak_t = _peak_t.sort_values(ascending=True)    # list is not sorted until now
        _sorted_peak_t.dropna(inplace=True)                     # if there are 'empty' NaN, remove them
        
        self.extPa["tPeaks"] = _sorted_peak_t
        
        # pass in "external parameters" for the peak extraction via extPa
        self.gpd.setExternalParameters(self.extPa)
        
        # returns 1 (works like True) when accept() or 0 (we take for False) otherwise.
        accepted = self.gpd.exec_()
        
        # data from dialog is stored in the attributes of self.gpd
        
        if accepted:
            self.noPeaks = False

            print (self.gpd.pkextracted_by_set) #the whitelist
            
            self.peakResults.readInPeakDialogResults(self.gpd.pkextracted_by_set)
            
            #make 'save' buttons available
            self.savePSRBtn.setEnabled(True)
            self.save_baselined_ROIs_Btn.setEnabled(True)
            self.extractGroupsDialog_Btn.setEnabled(True)
        
        else:
            print ('Returned but not happily: self.gpd.pkextracted_by_set is {}'.format(self.gpd.pkextracted_by_set))
        
        #ideas:
        #add set to the peaks combobox?!
        #accumulate histogram from individual ROI or store separately
        
    def plotNewData(self):
        """Do some setup immediately after data is loaded"""
        
        _sel_set = self.p3Selection.currentText()
        print ("Plot New Data with the p3 selector set for: ", _sel_set)
        y = {}
        
        self.p1.clear()
        self.p3.clear()
        
        for i, _set in enumerate(self.sheets):
            x = self.df[_set].index
            y[i] = self.df[_set].mean(axis=1).to_numpy()

            self.p1.plot(x, y[i], pen=(i,3))
        
            if _sel_set == _set:
                # curve
                self.p3.plot(x, y[i], pen=(i,3))
                
                xp, yp = self.peaksWrapper(x, y[i], _set)
                
                # need to add something to p3 scatter
                self.p3.plot(xp, yp, name="Peaks "+_set, pen=None, symbol="s", symbolBrush=(i,3))
                
                # create the object for parsing clicks in p3
                self.cA = clickAlgebra(self.p3)
                _p3_scatter = findScatter(self.p3.items)
                _p3_scatter.sigClicked.connect(self.clickRelay)
                _p3_scatter.sigPlotChanged.connect(self.manualUpdate)
        
        self.createLinearRegion()
        return
        
    def findSimplePeaks(self, xdat, ydat, name='unnamed'):
        """Simple and dumb peak finding algorithm"""
        # cut_off is not implemented here
        
        # SNR is used as a proxy for 'prominence' in the simple algorithm.
        self.cwt_SNR = self.cwt_SNR_Spin.value()
        
        peaks, _ = scsig.find_peaks(ydat, prominence=self.cwt_SNR)
        _npeaks = len(peaks)
        if _npeaks != 0:
            print ('Simple peak finding algorithm found {0} peaks in {1} trace with prominence {2}'.format(_npeaks, name, self.cwt_SNR))
            
            xp = xdat[peakcwt]
            yp = ydat[peakcwt]
            
        else:
            print ('No peaks found in {0} trace with simple algorithm with prominence {1}'.format(name, self.cwt_SNR))
            
            xp = []
            yp = []
           
        return xp, yp
        
        
    def findcwtPeaks(self, xdat, ydat, name='unnamed'):
        """Find peaks using continuous wavelet transform"""
        # think about storing per ROI peak data
        # indexes in peakcwt are not zero-biased
        self.cwt_width = self.cwt_w_Spin.value()
        self.cwt_SNR = self.cwt_SNR_Spin.value()
        peakcwt = scsig.find_peaks_cwt(ydat, np.arange(1, self.cwt_width), min_snr=self.cwt_SNR) - 1
        _npeaks = len(peakcwt)
        if _npeaks != 0:
            xpeak = xdat[peakcwt]
            ypeak = ydat[peakcwt]
            
            # filter out small peaks
            _cutOff = float (self.removeSml_Spin.value()) * ydat.max() / 100.0
            xpf = xpeak[np.where(ypeak > _cutOff)]
            ypf = ypeak[np.where(ypeak > _cutOff)]
            
            print ('wavelet transform peak finding algorithm found {0} peaks in {1} trace, width: {2}, SNR: {3}, cutOff: {4}.'.format(_npeaks, name, self.cwt_width, self.cwt_SNR, _cutOff))
        else:
            print ('No peaks found in {0} with cwt algorithm, width: {1}, SNR: {2}, cutOff: {4}.'.format(name, self.cwt_width, self.cwt_SNR, _cutOff))
            xpf = []
            ypf = []
        
        self.plots.peakslabel.setText("Number of peaks: {}".format(_npeaks))
        
        return xpf, ypf
    
    def manualUpdate(self):
        """Some editing was done in p3, so update other windows accordingly"""
        print ('Peak data in p3 changed manually')
       
        _sel_set = self.p3Selection.currentText()
        _ROI = self.ROI_selectBox.currentText()
        
        # update the peaks in p1 and histograms only
        remove_all_scatter(self.p1)
        
        #update p2 histograms
        self.updateHistograms()
        
        for i, _set in enumerate(self.sheets):
            #colours
            col_series = (i, len(self.sheets))
            
            if _sel_set == _set :
                _scatter = findScatter(self.p3.items)
                # sometimes a new scatter is made and this "deletes" the old one
                # retrieve the current manually curated peak data
                if _scatter is None:
                    print ('No Scatter found, empty data.')
                    xp = []
                    yp = []
                else:
                    xp, yp = _scatter.getData()
                
                # write peaks into results
                self.peakResults.addPeaks(_ROI, _sel_set, xp, yp)
                #print (self.peakResults.df[_ROI])
             
            xp, yp = self.peakResults.getPeaks(_ROI, _set)
            
            if self.split_traces:
                _target = self.p1stackMembers[i]
                # only one scatter item in each split view
                _t_scat = findScatter(_target.items)
                _t_scat.setData(xp, yp, brush=col_series)
                
            else:
                self.p1.plot(xp, yp, pen=None, symbol="s", symbolBrush=col_series)
            
            self.plots.peakslabel.setText("Number of peaks in {} set: {}".format(_set, len(yp)))
                
    def setBaselineParams (self):
        """Get parameters for auto baseline from GUI"""
        
        self.auto_bs_lam =  10 ** self.auto_bs_lam_slider.value()
        self.auto_bs_P =  10 ** (- self.auto_bs_P_slider.value() / 5)
    
    def peaksWrapper (self, x , y, set):
        """Simplify peak finding calls"""
        
        if self.simplePeaks:
            xp, yp = self.findSimplePeaks(x , y, name=set)
        else:
            xp, yp = self.findcwtPeaks(x , y, name=set)
    
        return xp, yp
        
        
    def ROI_Change(self):
        """General 'Update' method"""
        # called when ROI/trace is changed but
        # also when a new peak fit
        # approach is chosen.
        # consider renaming
        
        #### if baseline was changed in another window, all the peaks are now off...
        
        # we are not interested in updating data if there isn't any
        if self.dataLoaded == False:
            return
        
        # something changed in the control panel, get latest values
        _ROI = self.ROI_selectBox.currentText()
              
        if self.peak_CB.value() == 'simple':
            self.simplePeaks = True
        else:
            self.simplePeaks = False

        if self.autobs_Box.value() != 'None':
            self.auto_bs = True
            # populate values for automatic baseline removal from GUI
            if self.autobs_Box.value() == 'Auto':
                self.setBaselineParams()
            # print (self.auto_bs_lam_slider.value(), self.auto_bs_P_slider.value())
            
        else:
            self.auto_bs = False
            
        if self.SGsmoothing_CB.value() == 'On':
            # populate values for Savitsky-Golay smoothing from GUI
            self.sgSmooth = True
            self.sgWin = self.SGWin_Spin.value()
        else:
            self.sgSmooth = False

        # Empty the trace dictionary and the plots - perhaps we could be more gentle here?
        y = {}
        z = {}
        
        # Rather than doing this, need to keep the peak objects and set their data anew?
        self.p1.clear()
        
        # Rather than clearing objects in p3, we set their data anew
        _p3_items = self.p3.items
        _p3_scatter = findScatter(_p3_items)
        _p3_curve = findCurve(_p3_items)
        #print (_p3_items, _p3_scatter)
        
        for i, _set in enumerate(self.sheets):
            col_series = (i, len(self.sheets))
            x = np.array(self.df[_set].index)
            
            if _ROI == "Mean":
                y[i] = self.df[_set].mean(axis=1).to_numpy()
                
            elif _ROI == "Variance":
                y[i] = self.df[_set].var(axis=1).to_numpy()
                # we never want to subtract the steady state variance
                self.auto_bs = False
                print ('No baseline subtraction for variance trace')
                
            else:
                y[i] = self.df[_set][_ROI].to_numpy()
            
            if self.auto_bs:
                # baseline
                z[i] = baseline_als(y[i], lam=self.auto_bs_lam, p=self.auto_bs_P, niter=10)
                
                # subtract the baseline
                y[i] = y[i] - z[i]
                
                # plotting is done below
                
                
                
            if self.sgSmooth:
                print ('Savitsky Golay smoothing with window: {0}'.format(self.sgWin))
                y[i] = savitzky_golay(y[i], window_size=self.sgWin, order=4)

            if self.autoPeaks:
                
                #call the relevant peak finding algorithm
                xp, yp = self.peaksWrapper(x, y[i], _set)
                
                #write autopeaks into results
                self.peakResults.addPeaks(_ROI, _set, xp, yp)
            #
            else:
                #read in peak data from results (might be empty if it's new ROI)
                xp, yp = self.peakResults.getPeaks(_ROI, _set)
                
                if len(xp) == 0:
                    # Even though we are in manual peaks, the ROI was changed and there is no fit data.
                    print ("Peak results are empty, running auto peaks for {0} {1}".format(_ROI, _set))
                    
                    xp, yp = self.peaksWrapper(x, y[i], _set)
                    
                else:
                    # we are in manual peaks and there was data, so don't overwrite it
                    print ("Retrieved: ", xp, yp)
            
            # draw p1 traces and scatter
            if self.split_traces:
                target = self.p1stackMembers[i]
                target.clear()
                target.plot(x, y[i], pen=col_series)
                target.plot(xp, yp, pen=None, symbol="s", symbolBrush=col_series)
            else:
                self.p1.plot(x, y[i], pen=col_series)
                self.p1.plot(xp, yp, pen=None, symbol="s", symbolBrush=col_series)
                
                #plot baseline, offset by the signal max.
                self.p1.plot(x, z[i]-y[i].max(), pen=(255,255,255,80))
            
    
            #p3: plot only the chosen trace
            if self.p3Selection.currentText() == _set:
                if _p3_scatter is None:
                    # Do something about it, there are no peaks!
                    self.p3.addPlot(xp, yp, pen=None, brush=col_series)
                else:
                    _p3_scatter.clear()
                    _p3_scatter.setData(xp, yp, brush=col_series)
                
                _p3_curve.clear()
                _p3_curve.setData(x, y[i], pen=col_series)
                
        self.createLinearRegion()
        
        self.updateHistograms()
        
        return
        
    def setRanges(self):
        """ Collect the extremities of data over a set of sheets """
        self.ranges = {}
        self.ranges['xmin'] = self.df[self.sheets[0]].index.min()
        self.ranges['xmax'] = self.df[self.sheets[0]].index.max()
        self.ranges['ymin'] = self.df[self.sheets[0]].min().min()
        self.ranges['ymax'] = self.df[self.sheets[0]].max().max()
        
        #lazily compare across all sheets
        for sheet in self.df.values():
            if sheet.min().min() < self.ranges['ymin']:
                self.ranges['ymin'] = sheet.min().min()
            if sheet.max().max() > self.ranges['ymax']:
                self.ranges['ymax'] = sheet.max().max()
                
            if sheet.index.min() < self.ranges['xmin']:
                self.ranges['xmin'] = sheet.index.min()
            if sheet.index.max() > self.ranges['xmax']:
                self.ranges['xmax'] = sheet.index.max()
        return
    
    def save_peaks(self):
        print ("save_peak data and optionally histograms")
        
        #format for header cells.
        self.hform = {
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D5D4AC',
        'border': 1}
        
        if self.noPeaks:        #nothing to save
            print ('Nothing to save, no peaks found yet!')
            return
        self.filename = QFileDialog.getSaveFileName(self,
        "Save Peak Data", os.path.expanduser("~"))[0]
        
        #from XlsxWriter examples, John McNamara
        if self.filename:
            with pd.ExcelWriter(self.filename) as writer:
                
                # combine whitelist and blacklist dictionaries for output
                _output = {**self.gpd.pkextracted_by_set, **self.gpd.blacklisted_by_set}
                
                for _set in _output:
                    # in case there are duplicate peaks extracted, remove them and package into dummy variable
                    # [not the duplicates]
                    _pe = _output[_set].loc[~_output[_set].index.duplicated(keep='first')] #StackOverflow 13035764
                    
                    #skip the first row
                    _pe.to_excel(writer, sheet_name=_set, startrow=1, header=False)
                    
                    _workbook  = writer.book
                    _worksheet = writer.sheets[_set]
                    
                    #write header manually so that values can be modified with addition of the sheet (for downstream use)
                    header_format = _workbook.add_format(self.hform)
                    for col_num, value in enumerate(_pe.columns.values):
                        _worksheet.write(0, col_num + 1, value + " " +_set, header_format)
                
                if self.saveHistogramsOption:
                    self.save_histograms(writer)

    def save_histograms(self, writer):
        """write out histograms for each ROI to disk"""
        
        self.doHistograms()
        print (self.hDF.df.head(5))
        #save histograms into new sheet
        self.hDF.df.to_excel(writer, sheet_name="histograms",startrow=1, header=False)
        _workbook  = writer.book
        _worksheet = writer.sheets["histograms"]
        header_format = _workbook.add_format(self.hform)
        for col_num, value in enumerate(_pe.columns.values):
            _worksheet.write(0, col_num + 1, value + " hi", header_format)
            
    def save_baselined(self):
        
        # No filtering so far
        print ("save_baselined data")
        self.filename = QFileDialog.getSaveFileName(self,
        "Save Baselined ROI Data", os.path.expanduser("~"))[0]
        
    def open_file(self):
        """Open a dialog to provide sheet names"""
        
        self.filename = QFileDialog.getOpenFileName(self,
            "Open Data", os.path.expanduser("~"))[0]
        
        if self.filename:
            #very simple and rigid right now - must be an excel file with sheets
            with pg.ProgressDialog("Loading sheets...", 0, len(self.sheets)) as dlg:
                
                for _sheet in self.sheets:
                    dlg += 1
                    try:
                        self.df[_sheet] = pd.read_excel(self.filename, sheet_name=_sheet, index_col=0)
                        print (self.df[_sheet].head())
                    except:
                        print ("Probably: XLDR error- no sheet named exactly {0}. Please check it.".format(_sheet))
                        self.sheets.remove(_sheet)
                    
        print ("Loaded following sheets: ", self.sheets)
        
        self.ROI_list = ["Mean", "Variance"]
        #print (self.ROI_list)
        self.ROI_list.extend(self.df[self.sheets[0]].columns.tolist())
        
        #find out and store the size of the data
        self.setRanges()
        
        #split trace layout can be made now we know how many sets (conditions) we have
        self.createSplitTraceLayout()
        
        #populate the combobox for choosing which ROI to show
        self.ROI_selectBox.clear()
        self.ROI_selectBox.addItems(self.ROI_list)
        
        #populate the combobox for choosing the data shown in the zoom view
        self.p3Selection.clear()
        self.p3Selection.addItems(self.sheets)
        
        #create a dataframe for peak measurements
        self.peakResults = Results(self.ROI_list, self.sheets)
        print ("peakResults object created", self.peakResults, self.ROI_list)
        
        self.plotNewData()
        
        #updates based on GUI can now happen painlessly
        self.dataLoaded = True
        self.ROI_Change()
        


if __name__ == "__main__":
    #change menubar name from 'python' to 'SAFT' on macOS
    #from https://stackoverflow.com/questions/5047734/
    if sys.platform.startswith('darwin'):
    # Python 3: pip install pyobjc-framework-Cocoa is needed
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            if bundle:
                app_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                if app_info:
                    app_info['CFBundleName'] = app_name.upper() #ensure it is in upper case.
        except ImportError:
            print ("Failed to import NSBundle, couldn't change menubar name." )
            pass
    
    __version__ = "v. 0.2"
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

