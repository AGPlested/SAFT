import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import pandas as pd
from scipy.stats import binom, poisson

def cdf(x, func, max, step, *fargs):
    """calculate cdf for function. extra arguments (after x) for func in should be given in fargs
    func is the abitrary function to calculate the cdf"""
    
    #print ("Len (x) = {}".format(len(x)))
    #print (x, step)
    _denom = 0
    for d in np.arange(0, max, step):
        _denom += func(d, *fargs)
    
    f = []
    for xi in x:
        _num = 0
        #print ("xi: {} step: {}".format(xi, step))
        for d in np.arange(0, xi, step):
            _num += func(d, *fargs)
        
        f.append(_num / _denom)
    
    f_array = np.array(f)
    #print (f_array)
    return f_array
    

def gaussian(x, height, center, width, offset):
    """x is an array or a scalar"""
    return 0.399 * height / width * np.exp(-(x - center)**2 / (2 * width ** 2)) + offset

def nprGaussians(x, n, q, widths, scale, pr):
    """heights come from binomial (Pr) and an optimised scale parameter (number of events)"""
    g = gaussian (x, 0, 0, 1, 0) # create a blank in the correct x
    for k in range(n+1):
        b = binom.pmf(k, n, pr)
        g += gaussian(x, b * scale, k * q, widths, 0)
        #print ("Binomial k {}, n {}, pr {} = {}. q {}, w {}, scale {}".format(k, n, pr, b, q, widths, scale))
        
    return g

def fit_PoissonGaussians_global(num, q, ws, hy, hx, fixedW=False):
    # hy and hx are matrixes of n columns for the n histograms
    #print (hy.shape)
    nh = hy.shape[1]        # how many columns = how many functions
   
    mu = np.full(nh, 2)           # mean release rate, no bound (will be bounded 0 to num)
    _scale = 10                   # a scale factor depending on total no. of events measured
    
    if fixedW==False:
        guesses = np.array([q, ws, _scale, *mu])
           
        l_bounds = np.zeros (nh + 3)
        u_bounds = np.concatenate((np.full((3), np.inf), np.full(nh, num) ))
        return optimize.least_squares(globalErrFuncP, guesses, bounds = (l_bounds, u_bounds),
                                        args=(num, nh, hx.flatten(), hy.flatten()))
    else:
        guesses = np.array([q, _scale, *mu])
        l_bounds = np.zeros (nh + 2)
        u_bounds = np.concatenate((np.full((2), np.inf), np.full(nh, num) ))        #maximum value of mu is num
        return optimize.least_squares(globalErrFuncPW, guesses, bounds = (l_bounds, u_bounds),
                                        args=(num, ws, nh, hx.flatten(), hy.flatten()))

def poissonGaussians(x, n, q, widths, scale, mu):
    """Heights come from poisson with mean mu and an optimised scale parameter (no. of events)"""
    g = gaussian (x, 0, 0, 1, 0) # create a blank
    for k in range(n):
        b = poisson.pmf(k, mu)
        g += gaussian(x, b * scale, k * q, (k+1) * widths, 0)
        #print ("Poisson k {}, n {}, mu {} = {}. q {}, w {}, scale {}".format(k, n, mu, b, q, widths, scale))
    
    return g
   
   
def globalErrFuncPW(pa, num, ws, nh, hx, hy):
    """global poisson stats fit with fixed ws"""
    # 1-D function so hx and hy are passed flat
    # assume that pa is a list.
    _errfunc_list = []
    _hxr = hx.reshape(-1, nh)       # rows are inferred
    _hyr = hy.reshape(-1, nh)

    _q = pa[0]
    _scale = pa[1]

    # loop for each column
    for i in range(nh):
        _hx = _hxr[:, i]
        _hxc = np.mean(np.vstack([_hx[0:-1], _hx[1:]]), axis=0)
        
        # pa[i+2] is the relevant mu
        _e_i = (poissonGaussians(_hxc, num,  _q, ws, _scale, pa[i+2]) - _hyr[:, i])**2
        _errfunc_list.append(_e_i)

    return np.concatenate(_errfunc_list)     # FLAT -works for unknown n
   
def globalErrFuncP(pa, num, nh, hx, hy):
    """global mulit-gaussian fit with poisson stats"""
    # 1-D function so hx and hy are passed flat
    # assume that pa is a list...
    
    _errfunc_list = []
    _hxr = hx.reshape(-1, nh)       # rows are inferred
    _hyr = hy.reshape(-1, nh)

    _q = pa[0]
    _ws = pa[1]
    _scale = pa[2]

    # loop for each column
    for i in range(nh):
        _hx = _hxr[:, i]
        _hxc = np.mean(np.vstack([_hx[0:-1], _hx[1:]]), axis=0)
        
        # pa[i+3] is the relevant mu
        _e_i = (poissonGaussians(_hxc, num,  _q, _ws, _scale, pa[i+3]) - _hyr[:, i])**2
        _errfunc_list.append(_e_i)

    return np.concatenate(_errfunc_list)     #FLAT -should work for unknown n
    
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

def globalErrFuncBW(pa, num, ws, nh, hx, hy):
    """global binomial stats fit with fixed ws"""
    # 1-D function so hx and hy are passed flat
    # assume for now that pa is a list... it should be!
    _errfunc_list = []
    _hxr = hx.reshape(-1, nh)       # rows are inferred
    _hyr = hy.reshape(-1, nh)
    
    _q = pa[0]
    _scale = pa[1]
    
    # loop for each column
    for i in range(nh):
        _hx = _hxr[:, i]
        _hxc = np.mean(np.vstack([_hx[0:-1], _hx[1:]]), axis=0)
        # pa[i+2] is the relevant Pr
        _e_i = (nprGaussians(_hxc, num,  _q, ws, _scale, pa[i+2]) - _hyr[:, i])**2
        _errfunc_list.append(_e_i)

    return np.concatenate(_errfunc_list)     #FLAT -should work for unknown n


def globalErrFuncB(pa, num, nh, hx, hy):
    # 1-D function so hx and hy are passed flat
    # assume for now that pa is a list... it should be!
    _errfunc_list = []
    _hxr = hx.reshape(-1, nh)       # rows are inferred
    _hyr = hy.reshape(-1, nh)
    
    _q = pa[0]
    _ws = pa[1]
    _scale = pa[2]
    
    # loop for each column
    for i in range(nh):
        _hx = _hxr[:, i]
        _hxc = np.mean(np.vstack([_hx[0:-1], _hx[1:]]), axis=0)
        # pa[i+3] is the relevant Pr
        _e_i = (nprGaussians(_hxc, num,  _q, _ws, _scale, pa[i+3]) - _hyr[:, i])**2
        _errfunc_list.append(_e_i)

    return np.concatenate(_errfunc_list)     #FLAT -should work for unknown n

def fit_nprGaussians_global(num, q, ws, hy, hx, fixedW=False):
    # hy and hx are matrixes of n columns for the n histograms
    
    nh = hy.shape[1]        # how many columns = how many functions
    #print (hy.shape, nh)
    #l = np.arange(nh, dtype=np.double)
    pr = np.full(nh, 0.5)           # release probabilities (will be bounded 0 to 1)
    _scale = 10                         # a scale factor depending on no. of events measured
    
    if fixedW==False:
        guesses = np.array([q, ws, _scale, *pr])
    
        l_bounds = np.zeros (nh + 3)
        u_bounds = np.concatenate((np.full((3), np.inf), np.ones (nh)))
        return optimize.least_squares(globalErrFuncB, guesses, bounds = (l_bounds, u_bounds),
                                                    args=(num, nh, hx.flatten(), hy.flatten()))
    else:
        guesses = np.array([q, _scale, *pr])
    
        l_bounds = np.zeros (nh + 2)
        u_bounds = np.concatenate((np.full((2), np.inf), np.ones (nh)))
        return optimize.least_squares(globalErrFuncBW, guesses, bounds = (l_bounds, u_bounds),
                                                    args=(num, ws, nh, hx.flatten(), hy.flatten()))
    
def fit_nprGaussians (num, q, ws, hy, hx):
    # with fixed number of gaussians, q, ws
    
    _scale = 10         # a scale factor depending on no. of events measured
    pr = 0.5            # release probability (will be bounded 0 to 1)
    guesses = np.array([_scale, pr])
    
    errfunc = lambda pa, x, y: (nprGaussians(x, num, q, ws, *pa) - y)**2
    return optimize.least_squares(errfunc, guesses, bounds = ([0,0], [np.inf, 1]), args=(hx, hy))

def PoissonGaussians_display (hx, num, q, ws, optix):
    """oversample the Gaussian functions for a better display"""
    
    # optix being a 2-list or a 2-array, the x attribute of opti (from optimise). Scale, mu?
    # the ratio of the G. width to the histogram bin width tells us how much to oversample
    
    oversam = int(10 * (hx[1]-hx[0]) / ws)
    if oversam == 0:
        oversam = 2
    hx_u = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)
    hy_u = poissonGaussians(hx_u, num, q, ws, *list(optix))
    return hx_u, hy_u

def nprGaussians_display (hx, num, q, ws, optix, verbose=False):
    """oversample the Gaussian functions for a better display"""
    
    # optix being a 2-list or a 2-array, the x attribute of opti (from optimise).
    # the ratio of the G. width to the histogram bin width tells us how much to oversample
    
    oversam = int(10 * (hx[1]-hx[0]) / ws)
    if oversam == 0:
        oversam = 2
        
    if verbose: print ("nprGaussians_display", num, q, ws, optix, oversam)
    hx_o = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)
    hy_o = nprGaussians(hx_o, num, q, ws, *list(optix))
    return hx_o, hy_o
    
def nGaussians_display (hx, num, optix, verbose=False):
    """oversample the Gaussian functions for a better display"""
    
    # optix being a 2-list, the x attribute of opti (from optimise).
    # the ratio of the G. width to the histogram bin width tells us how much to oversample
    
    oversam = int(10 * (hx[1]-hx[0]) / optix[1])
    if oversam == 0:
        oversam = 2
        
    if verbose: print ("nGaussians_display", num, optix, oversam)
    hx_o = np.linspace(0, hx[-1], len(hx)*oversam, endpoint=False)
    hy_o = nGaussians(hx_o, num, *list(optix))
    return hx_o, hy_o
    
    
    
    
    
if __name__ == "__main__":
    
    # trial code
    
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
    opti = fit_poissGaussians_global(num, q, ws, hy, hx)
    #opti = fit_nprGaussians(num, q, ws, hy, hx)
    
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
    plt.ylabel("No. of events")
    plt.xlabel("dF/F")
    plt.legend(loc='upper right')

    plt.annotate(fitp,xy=(.85, .65), xycoords='figure fraction',
    horizontalalignment='right', verticalalignment='top',
    fontsize=10)

    #plt.show()

    plt.savefig('res{}.pdf'.format(num))
