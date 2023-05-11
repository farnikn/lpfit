import numpy as np

def errCovMat(Nreal, boxsize, k, Pk, bias, shotnoise, Sigma=None, r=None, bin_width=None, rbin_min=None, rbin_max=None):
    if r is None:
        bins = np.arange(rbin_min, rbin_max+bin_width, bin_width)
        r = 0.5*(bins[1:]+ bins[:-1])
    else:
        dr = np.mean(np.diff(r))
        rbin_min, rbin_max = r.min() - dr/2, r.max() + dr/2
        bins = np.arange(rbin_min, rbin_max+bin_width, bin_width)
        
    nr = len(r)
    covmat = np.zeros((nr, nr))
    
    b2 = bias**2.
    if Sigma is not None:
        Px = b2 * Pk * np.exp(-k**2 * Sigma**2)
    else:
        Px = b2 * Pk
        
    sigma_P2_cov = (Px + shotnoise)**2.
    for i in range(nr):
        for j in range(nr):
            # Pre-factor in front
            factor = 2./(2.*np.pi**2.*Nreal*boxsize**3.) * 3./(bins[i+1]**3. - bins[i]**3.) * 3./(bins[j+1]**3. - bins[j]**3.)
        
            # Four contribution from bin edges
            c1 =  bins[i+1]**2. * bins[j+1]**2. * np.trapz(sigma_P2_cov * spherical_jn(1, k*bins[i+1]) * spherical_jn(1, k*bins[j+1]) * k, x=np.log(k))
            c2 = -bins[i+1]**2. * bins[j]**2. * np.trapz(sigma_P2_cov * spherical_jn(1, k*bins[i+1]) * spherical_jn(1, k*bins[j]) * k, x=np.log(k))
            c3 = -bins[i]**2. * bins[j+1]**2. * np.trapz(sigma_P2_cov * spherical_jn(1, k*bins[i]) * spherical_jn(1, k*bins[j+1]) * k, x=np.log(k))
            c4 =  bins[i]**2. * bins[j]**2. * np.trapz(sigma_P2_cov * spherical_jn(1, k*bins[i]) * spherical_jn(1, k*bins[j]) * k, x=np.log(k))
        
            covmat[i,j] = factor * (c1 + c2 + c3 + c4)
        
    return covmat