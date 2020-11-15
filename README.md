# SAFT : Semi Automatic Fluorescent Trace analysis
v. 0.4  2020-11-15

SAFT was written in Python 3 for macOS. In principle it is cross-platform, in practice it was never launched on Windows or Linux. If you happen to get it working on either of these platforms, please let us know. 

### Getting Started on macOS (for Python/UNIX beginners) 

(Python/UNIX experts [see below](#experts))

Clone the SAFT repository to your machine by clicking the green "Code" button at the top of this page and selecting "Download Zip".

You will receive a directory called "SAFT-master", probably in the 'Downloads' folder. 

To get all the Python packages you need to run SAFT, we recommend using miniconda. Download `miniconda for python 3.6 bash` ; we provide everything you need after that. Here's the link:
https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html

Use Spotlight (Cmd+Space) to find and launch 'Terminal'. In the following, you will need to enter `commands` in the Terminal. You must be exact...

To install miniconda, you need to navigate to the directory where it downloaded, probably again 'Downloads'.

> `cd Downloads`

Then you can type the following command to make the installation. Use the space bar to page through the licence agreement. Answer (type) 'yes' to everything...

> `bash Miniconda3-latest-MacOSX-x86_64.sh`

(Free UNIX tip: if you start typing `bash Min` and then press the <Tab> key "->|", it will autocomplete)

After this you'll need to restart the Terminal window. 

Now you need to navigate to the SAFT directory *in the Terminal*. You start in your "Home" folder (its name is your username).  
The command you need is cd (change directory):

> `cd Downloads\SAFT-master`

To prepare for SAFT, get Conda to create the SAFT environment:
>`conda env create --file SAFTenv.yml`

Once all the packages have downloaded and Conda has arranged them all, to use them you need to switch to this environment: 

> `conda activate SAFT`

To run SAFT, type the following command into the Terminal:

> `pythonw SAFT.py`

On the first run in a freshly installed environment, you will have to wait for some of the packages. Matplotlib is kind enough to note that it is spending the time installing fonts.

From the window, you can open the example file `SAFT_Data.xlsx` for analysis. Read `instructions.pdf` for how to use the program.

You can also launch quantal fitting of histograms as a standalone app. If you add `test` some example data (the file `ExPeak_Data.xlsx`) will load automatically.

>`pythonw fitHistograms.py`
>`pythonw fitHistograms.py test`

### Experts
---

SAFT depends on quite a few packages, for its GUI (Qt, pyqtgraph, pyside2, matplotlib, pyobjc-framework-Cocoa), for number crunching (numpy, scipy, pandas) and for data handling (pandas, xlrd, openpyxl). Getting all these packages aligned is not easy. You can use anaconda, pip or compile from source. It is best to start with a fresh environment.

Conda currently provides Qt 5.9.7 and this makes it convenient to use conda to set up the package environment. To make getting the right environment in place easy, we include a conda environment .yml file in the distribution (SAFTenv.yml). To use this, you must have Anaconda or Miniconda. If you don't have Anaconda already, just get [Miniconda](# https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html).

To install the packages needed for SAFT into a fresh environment with Miniconda, go to the SAFT directory and enter the following in the Terminal,

> `conda env create --file SAFTenv.yml`

This will create an environment called 'SAFT' with a working set of packages.

As conda will tell you, to switch to this environment, enter the following,

>`conda activate SAFT`

then navigate to the SAFT directory and execute the commands as above to launch SAFT.

If you happen to already have an environment named SAFT, you would need to remove it first:

>`conda env remove --name SAFT`

**Very important** : Qt versions after 5.9 are not open source (breaking the conditions of our license), and coincidentally have serious bugs on macOS. 
We need Qt for the high-performance scientific plotting library [pyqtgraph](#http://www.pyqtgraph.org). 
PySide2 versions up to 5.15 seem to run happily with Qt 5.9.7 but it may be difficult to avoid a concurrent Qt update, unless you have a separate version of Qt compiled from open source and know how to point PySide2 at it. The latter solution is not stable. 

### Dependencies *current as of 2020-11-12*
---

Python 3.7.9   
qt==5.9.7   
PySide2>=5.9.0a1  
numpy==1.19.2 
scipy==1.5.2  
pyqtgraph==0.11.0  
pandas==0.24.0  
matplotlib==3.3.2  
xlrd==1.2.0  
openpyxl==3.0.5
pyobjc-framework-Cocoa==6.2.2
python.app 2  


