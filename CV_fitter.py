import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize


def pred_CV_squared(n, pr):
    """square of CV should be given by following formula"""
    ###CV^2 = (1 - pr[ca])/ (n * pr[ca])
    
    #vector size of pr
    return (1 - pr)/(n * pr)
   

def fit_CVs (N, CVs):
    """only CVs are fitted"""
    
 
    #print (N, CVs**2)
    #CVs =
        
    CVguesses = np.array([0.6, 0.3, 0.2])
    pr = np.array([0.1, 0.3, 0.4])

    errfunc = lambda pr: (pred_CV_squared(N, pr) - CVs**2)

    # loss="soft_l1" is bad!
    # should the limit of CV^2 estimates be 1?
    return optimize.least_squares(errfunc, CVguesses**2, bounds = (0, np.inf))


    
if __name__ == "__main__":
    
    # trial code
    
    mpl.rcParams['pdf.fonttype'] = 42

    #data = pd.read_csv('r47.txt', sep="\t", header=None)
    #data=data.as_matrix()
    #print (data)
    
    #x = data[:,0]
    #y = data[:,1]
    
    #x is [Ca] - not used for calculation
    x = np.array([0.5, 2, 4])
    y = np.array([0.60,.37,.16])
    
    for N in range(12):
  
        N = N + 1
        opti = fit_CVs(N, y)
    
        print (N, opti.x, opti.cost)
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
