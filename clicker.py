import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.spatial as scsp

#not used - this was for finding the closest point to the click
def do_kdtree(combined_x_y_arrays, points):
    mytree = scsp.cKDTree(combined_x_y_arrays)
    return mytree.query(points)

def coordinatesMat(twoArrays):
    #supplied with two arrays of x and y coordinates -output from scatter.getData()
    #returns N x 2 Numpy array of the x-y coordinates)
    xpts = twoArrays[0]
    ypts = twoArrays[1]
    return np.dstack([xpts.ravel(), ypts.ravel()])[0]
    
class clickAlgebra():
    def __init__(self, w, debug=False):
        # must pass the parent of the graph of interest
        self.dataClicked = False
        self.win = w
        self.debugging = debug
        print ('\nDebug- click algebra made with ', self.win, id(self))
    
    def onClick(self, *argv, dataLocked=True):
        # either a datapoint was clicked (remove it) or empty space (add a point)
        _db = self.debugging
        if _db: print ('\nDebug- onClick method of clickAlgebra. Before determination, self.dataClicked is {0}'.format(self.dataClicked))

        # work in PG ScatterPlotItem exclusively
        if len (argv) == 2:
            
            # set flag that data was clicked
            self.dataClicked = True
            if _db: print ('\nDebug- Point clicked, will be removed, self.dataClicked is {0}\n\n'.format(self.dataClicked))
            
            # when a data point is clicked, the point itself is sent (as argv[1])
            self.removePoint (argv[0], argv[1])
            
            return
       
        if self.dataClicked:
            # a single argument was passed
            # but a click from the plotitem comes after the data click
            # reset flag and get out
            if _db:
                print ('\nDebug- Escaping double call, self.dataClicked is {0}, resetting it to False.'.format(self.dataClicked))
            self.dataClicked = False
            
        else:
            # add a point because an empty part of the plot was clicked: len(argv) == 1
            self.addPoint(argv[0], dLocked=dataLocked)
            if _db: print ('\nDebug- Point added, self.dataClicked is {0}\n\n'.format(self.dataClicked))
            
            
    def addPoint(self, event, dLocked=True):
        _db = self.debugging
        # event is a scene.MouseClickEvent
        if _db: print ('\n\n**\nDebug- addPoint method with event: ', event)
        
        # get a list of the items under the click - includes scatter etc
        items = self.win.scene().items(event.scenePos())
        
        # pltitems is needed if we click outside existing scattter plot
        plt = [p for p in items if isinstance(p, pg.PlotItem)][0]
        pltitems = plt.items
        
        if _db:
            print ('\nDebug- scene().items:')
            for i, it in enumerate(items): print (i, it)
            
            print ('\nDebug- plotItems :')
            for p, pit in enumerate(pltitems): print (p,pit)
        
        # data to add the point to. take from items under the click!
        scatter = [s for s in items if isinstance(s, pg.ScatterPlotItem)]
        
        if len(scatter) == 1:
            #if we clicked within the bounds and got one scatter plot
            if _db: print ('\nDebug- Clicked in bounds of pg.ScatterPlotItem')
            scatter = scatter[0]
        else:
            # we clicked outside the bounds of the scatter data but still clicked on the plotdataitem
            # but we might have clicked the curve - so need to make sure we still grab the scatter
            if _db: print ('\nDebug- Clicked out-of-bounds of the pg.ScatterPlotItem')
            #hope that the following logic gets the right object!!!!
            PDIs = [d for d in pltitems if isinstance(d, pg.PlotDataItem)]
            if _db:
                print ('\nDebug- PlotDataItems :')
                for p, pd in enumerate(PDIs): print (p,pd)
            
            # seems extraneous
            # take the first one, might be wrong (that is, if we clicked on the curve data)
            # scatter = PDIs[0].scatter
            
            #try to improve by taking an object that instead has scatter points in it (the curve does not)
            for pdi in PDIs:
                x, _ = pdi.scatter.getData()
                if len(x) > 0:
                    
                    scatter = pdi.scatter
                    if _db: print ('Debug- Found scatter object with {} data points : {}'.format(len(x), scatter))
        #PG ScatterPlotItem
        if _db: print ('Debug- ScatterPlotItem object: ', scatter)
        
        # put coordinates from scatter plot into 2D Numpy array
        combined = coordinatesMat(scatter.getData())
        #print ("Before: (x,y)", combined)
        
        # map the point from the mouse click to the data
        if self.win.sceneBoundingRect().contains(event.scenePos()):
            mousePoint = self.win.vb.mapSceneToView(event.scenePos())
            
            if dLocked:
                
                # we want clicks locked to the trace: find the relevant curve data
                PCIs = [d for d in items if isinstance(d, pg.PlotCurveItem)]
                
                if _db:
                    #pop up a window to show the curve data being used
                    self.wx = pg.GraphicsLayoutWidget(show=True, title="Debugging")
                    self.wx.resize(600,190)
                    dwp = {} # to store the plots
                
                n = 0
                
                if len(PCIs) == 0:
                    # we clicked outside
                    
                    PDIs = [d for d in pltitems if isinstance(d, pg.PlotDataItem)]
                   
                    if _db:
                        print ('\nDebug- No PCIs found under click: ', PCIs)
                        print ('\nDebug- clicked outside bounds')
                        print ('\nDebug- PlotDataItems for finding curve :')
                        for p, pd in enumerate(PDIs): print (p,pd)
                               
                    for n, pdi in enumerate(PDIs):
                        trial_x, _ = pdi.curve.getData()
                        if len(trial_x) > 0:
                            _c = pdi.curve
                            
                            sx, sy = _c.getData()
                            if _db:
                                print ('Debug- {} Found PCurveItem with {} data pts. : {}'.format(n, len(trial_x), _c))
                                dwp[n] = self.wx.addPlot(title=str(n)+" "+str(type(_c)), y=sy, x=sx, pen="r")
                
                elif len(PCIs) > 0:
                    # we caught at least one above
                    for n, pci in enumerate(PCIs):
                        if _db: print ('\nDebug- PCIs found, iteration {}, {}'.format(n, pci))
                        trial_x, _ = pci.getData()
                        
                        if len(trial_x) > 0:
                            _c = pci
                            sx, sy = _c.getData()
                            if _db:
                                print ('\nDebug- Curve object: ', n, _c, type(_c))
                                dwp[n] = self.wx.addPlot(title=str(n)+" "+str(type(_c)), y=sy, x=sx, pen="y")
                        
                if _db:
                    dwp[n+1] = self.wx.addPlot(title="m_pt", x=[mousePoint.x()], y=[mousePoint.y()], pen=None, symbol='t')
                    #print ('\nDebug- debugging plot dictionary: ', dwp)
                
                idx = np.abs(sx - mousePoint.x()).argmin()
                
                ad_x = sx[idx]
                ad_y = sy[idx]
                
                if _db:
                    print ('\nDebug- dlocked is {}, using index of x data {}'.format(dLocked, idx))
                    dwp[n+2] = self.wx.addPlot(title="locked d_pt", x=[ad_x], y=[ad_y], pen=None, symbol='o')
            else:
                if _db: print ('\nDebug- Free clicking.')
                ad_x = mousePoint.x()
                ad_y = mousePoint.y()
            
            pt = np.array([ad_x, ad_y])
            if _db: print ('\nDebug- Mousepoint: {0}, {1}. Adding new data point: [x,y] {2}'.format(mousePoint.x(), mousePoint.y(), pt))
            
            # stick the new point into the data array
            added = np.append(combined, pt).reshape(-1,2)
            if _db: print ('\nDebug- scatter data after:\n', added)

            # update the plotted data
            scatter.setData(pos=added) #x=added[:,0], y=added[:,1]
            if _db: print ('\nDebug- New data in scatter: \n', coordinatesMat(scatter.getData()))
    
    def removePoint(self, scatter, k):
        _db = self.debugging
        # scatter is a PG ScatterPlotItem
        # a data point (k) was clicked so we are deleting it
        if _db: print ('\n\n**\nDebug- removePoint method with scatter and k:\n', scatter, k)
        
        # make a 2D array from the scatter data in the plot
        combined = coordinatesMat(scatter.getData())
        if _db: print ('\nDebug- Before: (x,y)\n', combined)
        
        # retrieved point from under the click
        pt = np.array([k[0].pos().x(), k[0].pos().y()])
        if _db: print ('\nDebug- Data point {0} clicked, to be removed '.format(pt))
        
        # mask point
        cleaned = combined[combined!=pt].reshape(-1,2)
        if _db: print ('\nDebug- After:\n', cleaned)
    
        # update the plotted data
        scatter.setData(pos=cleaned) #x=cleaned[:,0], y=cleaned[:,1]
        if _db: print ('\nDebug- New data in scatter:\n', coordinatesMat(scatter.getData()))
        

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication([])

    # Grid method; do others work too?
    # Passing a PlotItem seems to work as well.
    w = pg.GraphicsWindow()

    """
    # alternative trial code: THIS ALSO WORKS
    q = QtGui.QMainWindow()
    q.central_widget = QtGui.QWidget()
    q.central_layout = QtGui.QGridLayout()
    q.central_widget.setLayout(q.central_layout)

    q.setCentralWidget(q.central_widget)
    w = pg.GraphicsLayoutWidget(title="display")
    q.central_layout.addWidget(w)
    q.show()"""
    
    cA = clickAlgebra(w)

    for i in range(4):
        # dummy data
        x = np.random.normal(size=(i+1)*10)
        y = x*i + np.random.normal(size=(i+1)*10)
        
        # scatter plot
        k = w.addPlot(y=y*2,x=x*2, pen=None, symbol='o')
        # add a line plot to make sure it isn't affected by the clicked
        k.plot(x=x**2, y=y, pen="y")
        
        # dummy object - we just want to connect the data click signal
        sc = [d for d in k.items if isinstance(d, pg.PlotDataItem)][0]
        
        # connect each scatter to the onClick method
        sc.scatter.sigClicked.connect(cA.onClick) # to make sure the scatter object is passed

    # connect a click anywhere in the scene to the onClick method
    w.scene().sigMouseClicked.connect(cA.onClick)
    w.show()
   
   
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

