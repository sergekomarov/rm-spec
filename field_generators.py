from time import time
import numpy as np
from scipy import fftpack

'''
Functions to generate 2D/3D random fields and smooth models of galaxy clusters.
'''

def gen3d_smooth_model(shape, cluster_params, recovery_params):

    '''
    Generate a smooth beta-model for the radial distribution of the magnetic
    field component along the line of sight multiplied by the electron density:
    i.e., B_z(r)*n_e(r), in a galaxy cluster.

    Args:
        shape (tuple): (lx,ly) shape used to set the x-y dimensions of the model
        cluster_params (dict):  observed parameters of the cluster
        recovery_params (dict): other assumed parameters for the smooth model
    Returns:
        3d numpy array
    '''

    # central electron density
    ne0  = cluster_params['ne0']
    # cluster radius
    rc   = cluster_params['rc']
    # beta in the beta-model
    beta = cluster_params['beta']
    # pixel size in kiloparsecs
    kpc_px = cluster_params['kpc_px']

    # inclination of the radio source relative to the sky plane
    inclin = recovery_params['inclin']
    # B(r) = n_e(r)**alpha
    alpha  = recovery_params['alpha']
    # depth of the box in which the smooth model is generated
    lz = recovery_params['lz']

    lx,ly = shape
    # location of the cluster center on the sky in pixels
    ix0,iy0 = cluster_params['center']

    tn = np.tan(np.radians(90.-inclin))
    iz0 = int(ix0/(tn+1e-6))

    smod = np.zeros(shape)

    ix,iy,iz = np.meshgrid(np.arange(float(lx)),
                           np.arange(float(ly)),
                           np.arange(float(lz)), indexing='ij')
    r = np.sqrt((ix-ix0)**2 + (iy-iy0)**2 + (iz-iz0)**2) * kpc_px
    smod  = np.power(1+(r/rc)**2, -1.5*beta)
    smod  = np.power(smod, alpha+1) * ne0
    smod[ix-ix0>(iz-iz0)*tn] = 0.

    return smod


def gen2d(shape, p1=3.67, p2=3.67, kb=1e-1, C=1., msk=None, seed=0):

    '''
    Generate a 2D image with the given spectrum and the shape set by the mask.

    Args:
        shape (tuple): (lx,ly) image size in pixels
        p1: spectral slope above the knee (k<kb)
        p2: spectral slope below the knee (k>kb)
        kb: wavenumber of the knee
        C:  normalization constant
        msk:  mask (apply on top of the generated image)
        seed: seed for random number generator
    Returns:
        2D array: generated image with mask superimposed
    '''

    np.random.seed(seed)

    # First generate a larger image, then take a slice of the
    # original size to make the resulting image non-periodic.
    lx,ly=shape
    lx,ly = 2*lx,2*ly

    # form k-vectors
    kx,ky = np.meshgrid(np.arange(float(lx)), np.arange(float(ly)), indexing='ij')
    kx[kx>lx//2] = lx-kx
    ky[ky>ly//2] = ly-ky
    k = np.sqrt((kx/lx)**2 + (ky/ly)**2 + 1e-20)

    # set the spectral model
    z = np.sqrt( 1./(np.power(k/kb, p1) + np.power(k/kb, p2)) )
    z[0,0]=0.

    # inverse FFT
    cr = np.random.randn(lx,ly)
    ci = np.random.randn(lx,ly)
    ak = z * (cr + 1j * ci)
    norm = (z**2).sum()
    img = fftpack.ifftn(ak)
    img *= C * sqrt(lx*ly/norm)
    print( 'Checking the Parseval theorem for the generated 2D image:\n'+
        f'vf={np.sqrt((np.abs(ak)**2 / norm).mean())}, vr={np.sqrt((np.abs(img)**2).mean())}')

    # take a slice to get the same shape as given in the arguments
    lx,ly = lx//2,ly//2
    img = img[:lx,:ly].real
    # apply the mask
    if mask!=None: img[msk==0.]=np.NaN

    return img


def gen3d_divfree(shape, p1=3.67, p2=3.67, kb=1e-1, C=1., seed=0):

    '''
    Generate a single component of a 3D divergence-free vector field with the given spectrum.
    The spectral model in 3D is always 'soft_knee'.

    Args:
        shape (tuple): (lx,ly,lz) shape of the generated box
        p1: spectral slope above the knee (k<kb)
        p2: spectral slope below the knee (k>kb)
        kb: wavenumber of the knee
        C:  normalization constant
        seed: seed for random number generator
    Returns:
        3d numpy array
    '''

    print('Setting 3D harmonics...')

    np.random.seed(seed)

    # First generate a larger image, then take a slice of the
    # original size to make the resulting image non-periodic.
    lx,ly,lz = shape
    lx,ly,lz = 2*lx,2*ly,2*lz

    # form k-vectors
    kx,ky,kz = np.meshgrid(np.arange(float(lx)),
                           np.arange(float(ly)),
                           np.arange(float(lz)), indexing='ij')
    kx[kx>lx//2] = lx-kx
    ky[ky>ly//2] = ly-ky
    kz[kz>lz//2] = lz-kz
    k = np.sqrt((kx/lx)**2 + (ky/ly)**2 + (kz/lz)**2 + 1e-20)

    # set the spectral model
    z = np.sqrt( 1./(np.power(k/kb, p1) + np.power(k/kb, p2)) )
    z[0,0,0]=0.

    phr = 2*np.pi*np.random.random((lx,ly,lz))
    phi = 2*np.pi*np.random.random((lx,ly,lz))
    zr = 2*np.random.random((lx,ly,lz)) - 1
    zi = 2*np.random.random((lx,ly,lz)) - 1
    cr = np.random.randn(lx,ly,lz)
    ci = np.random.randn(lx,ly,lz)

    akxr = cr * np.cos(phr) * np.sqrt(1-zr*zr)
    akyr = cr * np.sin(phr) * np.sqrt(1-zr*zr)

    akxi = ci * np.cos(phi) * np.sqrt(1-zi*zi)
    akyi = ci * np.sin(phi) * np.sqrt(1-zi*zi)

    bkzi =  (kx*akyr - ky*akxr) #/k
    bkzr = -(kx*akyi - ky*akxi) #/k
    bkz = z/k * (bkzr + 1j*bkzi)
    norm = 2./3*(z**2).sum()

    print('Taking inverse 3D FFT...')

    out = fftpack.ifftn(bkz)
    out *= C*np.sqrt(lx*ly*lz/norm)
    print( 'Checking the Parseval theorem for the 3D field:\n'+
        f'vf={np.sqrt((np.abs(bkz)**2 / norm).mean())}, vr={np.sqrt((np.abs(out)**2).mean())}')

    # take a slice
    lx,ly,lz = lx//2,ly//2,lz//2
    return out[:lx,:ly,:lz].real
