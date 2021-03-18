import sys, os
from time import time
import numpy as np
from scipy import special, fftpack
import multiprocessing as mp
import configparser

from field_generators import gen2d, gen3d_divfree, gen3d_smooth_model
from utils import add_padding, data_fft, load_data, write_data


class RMspec:
    '''
    Implements the calculation of the power spectral density of an irregular
    image using Mexican-hat filtering.

    For astrophysical rotation measure maps, using some knowledge about
    the observed galaxy cluster, also allows deprojection into 3D to estimate
    the 3D PDS of fluctuating magnetic fields in the cluster.
    '''

    def __init__(self, input_path=None, output_dir='out', config_path='params.cfg'):

        '''
        Obtains parameters from the config file, loads the data.

        Args:
            input_path (str):  path to the input file, overrides the config file
            output_dir (str):  path to the output directory
            config_path (str): path to the configuration file
        '''

        # parse the config file
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
        except IOError:
            print('Error: No config file [params.cfg] found in the root directory')
            sys.exit()

        # Input (path to the input image)

        if input_path is None:
            self.input_path = config.get("input", "input_path")

        input_fname_parts = os.path.split(self.input_path)[-1].split('.')
        if len(input_fname_parts)==1:
            print('Error: the input file has no extension, please use .npy or .fits')
            sys.exit()
        input_fname=('.').join(input_fname_parts[:-1])  # file name w/o extension

        # output folder
        if not os.path.exists(output_dir): os.mkdir(output_dir)

        # output file names
        self.output_paths = {}
        out_files = ['deproj','smooth','mask', 'smooth3d']
        for f in out_files:
            self.output_paths[f] = os.path.join(output_dir,
                                                f+ '_'+input_fname+'.npy')
        self.output_paths['pds'] = os.path.join(output_dir,
                                                'pds_'+input_fname+'.txt')

        # '''
        # deproject=0: calculate the 2D power spectral density of the input image
        # deproject=1: deproject the image and estimate the 3D spectrum of the
        #              fluctuating magnetic fields
        # '''
        # self.regime = config.get("input", "regime")

        # Observed cluster parameters parameters (use only when need to deproject 3D PDS)

        # beta model parameters
        self.cluster_params = {}
        self.cluster_params['ne0'] = config.getfloat("cluster_params", "ne0")
        self.cluster_params['rc'] = config.getfloat("cluster_params", "rc")
        self.cluster_params['beta'] = config.getfloat("cluster_params", "beta")
        # size of pixel in kiloparsecs
        self.cluster_params['kpc_px'] = config.getfloat("cluster_params", "kpc_px")
        # Galactic background to subtract from the observed RM map
        self.cluster_params['gal_bg'] = config.getfloat("cluster_params", "gal_bg")
        # location of the cluster center on the sky in pixels
        self.cluster_params['center'] = [config.getint("cluster_params", "ix0"),
                                         config.getint("cluster_params", "iy0")]

        #recovery
        self.recovery_params = {}
        self.recovery_params['alpha'] = config.getfloat("recovery_params", "alpha")
        self.recovery_params['inclin'] = config.getfloat("recovery_params", "inclin")
        # depth of the 3D box used to generate the smooth model
        self.recovery_params['lz'] = config.getint("recovery_params", "Lz")

        #input
        # mask_path = config.get("input", "mask_path")
        #
        # #output
        # rm_out_path = config.get("output", "rm_out_path")
        #
        # if regime=="2d":
        #     #gen2d_params
        #     lx = config.getint("gen2d_params", "Lx")
        #     ly = config.getint("gen2d_params", "Ly")
        #     p1 = config.getfloat("gen2d_params", "p1")
        #     p2 = config.getfloat("gen2d_params", "p2")
        #     kb = config.getfloat("gen2d_params", "kb")
        #     C = config.getfloat("gen2d_params","C")
        #     apply_mask = config.getboolean("gen2d_params", "apply_mask")
        #     bw = config.getfloat("gen2d_params", "beam_width")
        #
        # elif regime=="3d":
        #     #gen3d_params
        #     lx = config.getint("gen3d_params", "Lx")
        #     ly = config.getint("gen3d_params", "Ly")
        #     lz = config.getint("gen3d_params", "Lz")
        #     p1 = config.getfloat("gen3d_params", "p1")
        #     p2 = config.getfloat("gen3d_params", "p2")
        #     kb = config.getfloat("gen3d_params", "kb")
        #     C = config.getfloat("gen3d_params", "C")
        #     apply_mask = config.getboolean("gen3d_params", "apply_mask")
        #     inclin = config.getfloat("gen3d_params", "inclin")
        #     alpha = config.getfloat("gen3d_params", "alpha")

        ftype = self.input_path.split('.')[-1].lower()

        # load data
        d = load_data(self.input_path, ftype=ftype, print_info=True)

        # crop it
        ir, jr = np.nonzero(np.invert(np.isnan(d)))
        imin, jmin, imax, jmax = ir.min(),jr.min(), ir.max(),jr.max()
        self.data = d[imin:imax+1, jmin:jmax+1]
        self.data_dim = self.data.shape

        # adjust the cluster center location after the croping
        self.cluster_params['center'][0] -= imin
        self.cluster_params['center'][1] -= jmin

        # make the mask given the cropped input data
        self.mask = np.array(np.invert(np.isnan(self.data)), dtype=float)

        # save the mask
        write_data(self.output_paths['mask'], self.mask, ftype='npy')

        # subtract the background
        self.data -= self.cluster_params['gal_bg']


    def deproject(self, cluster_params=None, recovery_params=None):

        kpc_px = self.cluster_params['kpc_px']
        lx,ly = self.data.shape

        print('generate a 3D smooth model of the cluster...')
        smod = gen3d_smooth_model(shape=(lx,ly),
                                  cluster_params=self.cluster_params,
                                  recovery_params=self.recovery_params)
        write_data(self.output_paths['smooth3d'], smod[:,ly//2,:], ftype='npy')

        print('deproject the image...')

        # integrate it along the lign of sight
        I0 = np.sqrt( (smod**2).sum(axis=2)) * kpc_px * 812.
        write_data(self.output_paths['smooth'], I0, ftype='npy')

        # divide the image by the smooth model
        ind = (I0!=0.) * np.invert(np.isnan(self.data))
        self.data[ind] /= I0[ind]
        write_data(self.output_paths['deproj'], self.data, ftype='npy')

    #    X,Y = meshgrid(arange(ly), arange(lx))
    #    circ = sqrt((Y-479)**2+(X-46)**2) < 225.
    #    rm[circ] = NaN

        print('deprojection done\n')


    def get_spectrum(self, ns=30, smin=3, smax=100, p_corr=0., nt=1):

        '''
        Calculate the isotropic power spectral density (PDS) for an image of
        irregular shape.

        Args:
            ns (int):   number of length-scales at which the PDS is calculated
            smin (int): minimum length-scale in pixels (>=2)
            smax (int): maximum length-scale in pixels
            p_corr: spectral slope if known beforehand (to correct normalization)
            nt:     number of threads for multiprocessing

        Returns:
            kr:     radial wavenumbers
            pds:    PDS measured at kr
        '''

        print('calculate the PDS...')

        # replace NaNs by zeros, add padding
        dp = add_padding(self.data, width=0.5)
        mp = add_padding(self.mask, width=0.5)

        # Fourier transform of the padded image and mask
        dpf, mpf = data_fft(dp), data_fft(mp)

        #pds = mp.Array('d', np.zeros(N))
        pds = np.zeros(ns)

        # set the range of radial wave numbers
        lx,ly = dp.shape
        krmin = 1./smax if smax<=0.5*max(lx,ly) else 0.5*max(lx,ly) # mind zero padding
        krmax = 1./smin       # 1/2 is the Nyquist frequency
        fk = (krmax/krmin)**(1./(ns-1))
        kr = krmin * np.power(fk, np.arange(ns))

        eps = 1e-3

        #t=time()

        # Split calculation for different wave numbers between threads

        params = [{'i':i, 'kri':kr[i], 'datf':dpf, 'mskf':mpf, 'msk':mp,
                 'eps':eps, 'p_corr':p_corr} for i in range(ns)]

        if nt>1:
            with mp.Pool(nt) as pool:
                pds = pool.map(calc_pds_single, **params)
            pool.join()
        else:
            pds = map(calc_pds_single, **params)

        #print 'execution time=',time()-t

        if np.any(np.isnan(pds)): print('Error: NaN in the PDS')

        self.scales = 1./kr
        self.pds = pds
        np.savetxt(self.output_paths['pds'], np.asarray([self.scales,self.pds]))
        print(f'PDS calculated and saved as {eps}')

        return (self.scales,self.pds)


#------------------------------------------------------------------

def calc_pds_single(i=0, kri=1e-1, datf=None, mskf=None, msk=None,
                    eps=1e-3, p_corr=0.):

    '''
    Calculate the PDS value for a single radial wavenumber kri=kr[i].

    Args:
        kri (float):            radial wavenumber
        datf, mskf (2d arrays): Fourier transforms of the padded image and mask
        msk (2d array):         padded mask
        eps (float):    sets the width difference between the two Gaussian filters
        p_corr (float): spectral slope if known beforehand (to correct normalization)
    Returns:
        float: PDS value at kri

    kri sets the width of the Mexican hat filter applied to the image.

    The Mexican hat is the difference of two Gaussian filters of similar
    width: G1 and G2.

    The PDS value is obtained from the total variance of the convolved image.
    '''

    (lx,ly) = datf.shape
    Ikr = np.zeros((lx,ly))

    # widths of the two Gaussian filters whose difference is the Mexican hat filter
    sigma = 0.225079/kri
    sigma1 = sigma/np.sqrt(1.+eps)
    sigma2 = sigma*np.sqrt(1.+eps)

    # Set Gaussian filters on the grid in k-space

    fG1 = lambda kx,ky: np.exp(-2 * np.pi**2 * (kx**2+ky**2) * sigma1**2)
    fG2 = lambda kx,ky: np.exp(-2 * np.pi**2 * (kx**2+ky**2) * sigma2**2)

    kx,ky = np.meshgrid(np.linspace(0,1.-1./lx,lx),
                        np.linspace(0,1.-1./ly,ly), indexing='ij')

    G1 = fG1(kx,ky)+fG1(1.-kx,ky)+fG1(kx,1.-ky)+fG1(1.-kx,1.-ky)
    G2 = fG2(kx,ky)+fG2(1.-kx,ky)+fG2(kx,1.-ky)+fG2(1.-kx,1.-ky)

    # Mexican hat filter power
    fpwr = (np.abs(G1-G2)**2).sum()

    if not np.all(msk==1.):

        # general case

        # multiply the data and mask by the filter in k-space
        # and take the inverse Fourier transform
        conv_dG1 = fftpack.ifftn(G1*datf)
        conv_dG2 = fftpack.ifftn(G2*datf)
        conv_mG1 = fftpack.ifftn(G1*mskf)
        conv_mG2 = fftpack.ifftn(G2*mskf)

        #print abs(datf).max()

        '''
        Divide the G1/G2 image convolutions by the respective mask convolutions
        and get the difference beteen G1 and G2. Division by the mask helps
        remove the spurious harmonics resulting from the sharp edges of
        the irregular image.
        '''
        nz = msk==1.
        Ikr[nz] =(conv_dG1[nz].real / conv_mG1[nz].real -
                  conv_dG2[nz].real / conv_mG2[nz].real)
    else:

        '''
        Case of a periodic rectangular image.
        No need to divide by the mask -- simlpy convolve with the Mexican hat.
        '''

        conv_dG = fftpack.ifftn((G1-G2) * datf)
        Ikr = conv_dG*np.sqrt(lx*ly)

        vr = np.sqrt((np.abs(Ikr)**2).mean())
        vf = np.sqrt((np.abs((G1-G2) * datf)**2).mean())
        print(f'Checking the Parseval theorem (backward FFT):\nvr={vr}\nvf={vf}')

    # total variance
    var = ( lx*ly / msk.sum() * (np.abs(Ikr)**2).sum() )

    # divide by the filter power
    pds_tilda = var / fpwr

    # correction coefficient in case the spectral slope is known beforehand
    # (used for testing on mock images to fix normalization)
    corr_f = (2**(p_corr/2) *
              special.gamma(3-p_corr/2)/
              special.gamma(3))

    return pds_tilda / corr_f

    #print 'kr='+str(kr)+' P(kr)='+str(pds[i])

    #make an example of the filtered image
    #if i == 20:
    #    figure()
    #    imshow(Ikr)
    #    savefig('../img/filt.ps')
    #    close()
    #    Ikr1 = trim(Ikr).copy()
    #    Ikr1[Ikr1==0.]=NaN
    #    write_fits('../img/filt.fits', Ikr1)
