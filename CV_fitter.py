import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize



def pred_CV_squared(n, pr):
    """square of CV should be given by following formula"""
    ###CV^2 = (1 - pr[ca])/ (n * pr[ca])

    return (1 - pr)/(n * pr)
   
def errFunc#(pa, num, ws, nh, hx, hy):
    #"""global poisson stats fit with fixed ws"""
    # 1-D function so hx and hy are passed flat
    # assume that pa is a list.
    
    # for each value of N, we calculate the three CVs
    _errfunc_list = []
    _hxr = hx.reshape(-1, nh)       # rows are inferred
    _hyr = hy.reshape(-1, nh)

    _q = pa[0]
    _scale = pa[1]

    # loop for each column
    for i in range(nCa):
        _hx = _hxr[:, i]
        _hxc = np.mean(np.vstack([_hx[0:-1], _hx[1:]]), axis=0)
        
        # pa[i+2] is the relevant mu
        _e_i = (poissonGaussians(_hxc, num,  _q, ws, _scale, pa[i+2]) - _hyr[:, i])**2
        _errfunc_list.append(_e_i)

    return np.concatenate(_errfunc_list)     # FLAT -works for unknown n

def fit_over_range_N ():

    for N_sites in range (Nmax):
       
    return False
    
def fit_CVs (N, Ca, CVs):
    """only CVs are fitted"""
    
    #[Ca]
    x = np.array([0.5, 2, 4])
    
    #CVs =
        
    CVguesses = np.array([0.2, 0.5, 0.8])

    #N_sites = N_sites + 1

    for CV in CVguesses:
        
        errfunc = lambda N, pr: (pred_CV_squared(N, pr) - y^2)**2

    # loss="soft_l1" is bad!
    # should the limit of CV^2 estimates be 1?
    return optimize.least_squares(errfunc, guesses, bounds = (0, np.inf), args=(N, pr))

    
def result_display (hx, num, optix, verbose=False):
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

    x = data[:,0]
    y = data[:,1]

    #there are no parameters 


    # single N right now
    opti = fit_CVs(N, x, y)
    
    print (opti)
    """
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
    """
