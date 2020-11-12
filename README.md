# SAFT v. 0.3
Semi Automatic Fluorescent Trace analysis

Launch SAFT from the Terminal with the command,

pythonw SAFT.py

From the window, you can open the SAFT_Data.xlsx file for analysis. 

On the first run in a freshly installed environment, you will have to wait for some of the packages, Matplotlib is kind enough to note that it is spending the time installing fonts.

You can also launch quantal fitting of histograms as a standalone app, and the following command will automatically load some example data

pythonw fitHistograms.py test



## Requirements
SAFT was written for macOS. In principle it is cross-platform, in practice it was never launched on Windows or Linux. 

SAFT depends on quite a few packages, for its GUI (Qt, pyqtgraph, pyside2, matplotlib, pyobjc-framework-Cocoa), for number crunching (numpy, scipy, pandas) and for data handling (pandas, xlrd, openpyxl). 

Getting all these packages aligned is not easy. You can use anaconda, pip or compile from source. 

Very important : Qt versions after 5.9 are not open source (breaks licensing conditions), and coincidentally have serious bugs on macOS. 
We need Qt for the high-performance scientific plotting library pyqtgraph. 
PySide2 versions up to 5.15 seem to run happily with qt 5.9.7 but it may be difficult to avoid a concurrent Qt update, unless you have a separate version of Qt compiled from open source and know how to point PySide2 at it. The latter solution is not stable. 

Conda currently provides Qt 5.9.7 and this makes it convenient to use conda to set up the package environment. To make getting the right environment in place, we include a conda environment .yml file in the distribution (SAFTenv.yml). To use this, you must have Anaconda or Miniconda. If you don't have Anaconda already, just get Miniconda. https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html

To install the packages into a fresh environment with Miniconda, go to the SAFT directory and enter the following in the Terminal:

conda env create --file SAFTenv.yml

This will create an environment called 'SAFT' with a working set of packages.

As conda will tell you, to use this environment, enter the following:

conda activate SAFT

If you happen to already have an environment named SAFT, you will need to remove it first:

conda env remove --name SAFT

Current dependencies
----------------------------
Python 3.7.x

qt==5.9.7

PySide2>=5.9.x

numpy==1.18.4

scipy==1.2.0

pyqtgraph==0.11.0

pandas==0.24.0

matplotlib (3.3.2 works?)

xlrd>=1.0.0

openpyxl

pyobjc-framework-Cocoa

python.app 3.x


