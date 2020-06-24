import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import pandas as pd



def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset
    
def nGaussians(x, n, spacing, widths, *heights):
    g = gaussian (x, 0, 0, 1, 0) # create a blank
    for j in range(n):
        g += gaussian(x, heights[j], (j+1) * spacing, widths, 0)
        
    return g
    
def fit_nGaussians (num, q, ws, hy, hx):

    h = np.random.rand(num) * np.average(hy) #guesses for heights

    guesses = np.array([q, ws, *h])

    errfunc = lambda pa, x, y: (nGaussians(x, num, *pa) - y)**2

    # loss="soft_l1" is bad!
    return optimize.least_squares(errfunc, guesses, bounds = (0, np.inf), args=(hx, hy))
    
    
def nGaussians_display (hx, num, p):
    oversam = int(10 * (hx[1]-hx[0]) / p.x[1]) # the ratio of the G. width to the histogram bin width tells us how much to oversample
    hx_u = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)   #oversample to get nice gaussians
    
    hy_u = nGaussians(hx_u, num, *list(p.x))
    return hx_u, hy_u
    
if __name__ == "__main__":

    mpl.rcParams['pdf.fonttype'] = 42

    data = pd.read_csv('r47.txt', sep="\t", header=None)
    data=data.as_matrix()
    #print (data)

    hx = data[:,0]
    hy = data[:,1]

    num = 5     # number of gaussians (will not be optimised
    q = .02   # quantal size
    ws = .01    # width of the gaussian

    opti = fit_nGaussians(num, q, ws, hy, hx)

    print (opti)

    plt.bar(hx, hy, color='orange', label='Peaks', width=(hx[1]-hx[0])*.95, alpha=0.4, edgecolor='black')

    #hx_u = np.linspace(0, hx[-1], len(hx)*10, endpoint=False)   #oversample to get nice gaussians
    fitp = ('q = {:.3f}\nw = {:.3f}'.format(opti.x[0], opti.x[1]))

    hx_u, hy_u = nGaussians_display(hx, num, opti)

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
