import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import pandas as pd
from scipy.stats import binom



def nprGaussians(x, n, q, widths, pr, scale):
    """heights come from binomial (pr) and an optimised scale parameter (number of events)"""
    g = gaussian (x, 0, 0, 1, 0) # create a blank
    for j in range(n):
        b = binom.pmf(j, n-1, pr)
        g += gaussian(x, b * scale, j * q, widths, 0)
        #print ("Binomial k {}, n {}, pr {} = {}. q {}, w {}, scale {}".format(j, n-1, pr, b, q, widths, scale))
        
    return g
    
def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2 / (2 * width ** 2)) + offset
    
def nGaussians(x, n, spacing, widths, *heights):
    g = gaussian (x, 0, 0, 1, 0) # create a blank
    for j in range(n):
        g += gaussian(x, heights[j], j * spacing, widths, 0)
        
    return g
    
def fit_nGaussians (num, q, ws, hy, hx):
    """heights are fitted"""
    h = np.random.rand(num) * np.average(hy) # array of guesses for heights

    guesses = np.array([q, ws, *h])

    errfunc = lambda pa, x, y: (nGaussians(x, num, *pa) - y)**2

    # loss="soft_l1" is bad!
    return optimize.least_squares(errfunc, guesses, bounds = (0, np.inf), args=(hx, hy))
    
def fit_nprGaussians (num, q, ws, hy, hx):
    # with fixed number of gaussians, q, ws
    
    pr = 0.5            # release probability (will be bounded 0 to 1)
    events = 10         # a scale factor depending on no. of events measured
    guesses = np.array([pr, events])
    
    errfunc = lambda pa, x, y: (nprGaussians(x, num, q, ws, *pa) - y)**2
    return optimize.least_squares(errfunc, guesses, bounds = ([0,0], [1, np.inf]), args=(hx, hy))
   
def nprGaussians_display (hx, num, q, ws, p):
    # oversample the Gaussian functions for a better display
    oversam = int(10 * (hx[1]-hx[0]) / ws) # the ratio of the G. width to the histogram bin width tells us how much to oversample
    hx_u = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)  
    hy_u = nprGaussians(hx_u, num, q, ws, *list(p.x))
    return hx_u, hy_u
    
def nGaussians_display (hx, num, p):
    # oversample the Gaussian functions for a better display
    oversam = int(10 * (hx[1]-hx[0]) / p.x[1]) # the ratio of the G. width to the histogram bin width tells us how much to oversample
    hx_u = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)
    hy_u = nGaussians(hx_u, num, *list(p.x))
    return hx_u, hy_u
    
if __name__ == "__main__":

    mpl.rcParams['pdf.fonttype'] = 42

    data = pd.read_csv('r47.txt', sep="\t", header=None)
    data=data.as_matrix()
    #print (data)

    hx = data[:,0]
    hy = data[:,1]

    #these parameters are not optimised (in nprgaussians)
    num = 8    # number of gaussians (will not be optimised
    q = .062   # quantal size
    ws = .015   # width of the gaussian

    # just a straight line at the moment.
    #opti = fit_nGaussians(num, q, ws, hy, hx)
    opti = fit_nprGaussians(num, q, ws, hy, hx)
    
    print (opti)

    plt.bar(hx, hy, color='orange', label='Peaks', width=(hx[1]-hx[0])*.95, alpha=0.4, edgecolor='black')

    #hx_u = np.linspace(0, hx[-1], len(hx)*10, endpoint=False)   #oversample to get nice gaussians
    #fitp = ('q = {:.3f}\nw = {:.3f}'.format(opti.x[0], opti.x[1]))
    fitp = ('Pr = {:.3f}'.format(opti.x[0]))
    #hx_u, hy_u = nGaussians_display(hx, num, opti)
    hx_u, hy_u = nprGaussians_display(hx, num, q, ws, opti)
    
    plt.plot(hx_u, hy_u,
       c='black', label='Fit of {} Gaussians'.format(num))
    plt.title("Optical quantal analysis of glutamate release")
    plt.ylabel("N events")
    plt.xlabel("dF/F")
    plt.legend(loc='upper right')

    plt.annotate(fitp,xy=(.85, .65), xycoords='figure fraction',
    horizontalalignment='right', verticalalignment='top',
    fontsize=10)

    #plt.show()

    plt.savefig('res{}.pdf'.format(num))
