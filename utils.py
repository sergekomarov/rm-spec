import sys
import numpy as np
from scipy import fftpack
import astropy.io.fits as pyfits

def get_last2d(data):
    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 2)
    slc += [slice(None), slice(None)]
    return data[slc]

def load_data(fname, ftype='fits', print_info=True):

    '''
    Load an image from a .fits or .npy file.
    An image of irregular shape is assumed to be surrounded by NaN pixels.

    Args:
        fname (str):        path to the input file
        ftype (str):        'fits' or 'npy'
        print_info (bool):  print .fits file info
    Returns:
        2D numpy array
    '''

    if ftype=='fits':

        with pyfits.open(fname) as hdulist:

            if print_info:
                print(str(hdulist.info()))
            if len(np.shape(hdulist[0].data))>2:
                data = get_last2d(hdulist[0].data)
            else:
                data = hdulist[0].data

    elif ftype=='npy':
        data = np.load(fname)
    else:
        print('Error: unknown input file type, please use .npy or .fits ')
        sys.exit()

    # check array dimensions
    if len(data.shape)!=2:
        print(f'Error: wrong shape of the input array, got {data.shape}')
        sys.exit()

    return data


def write_data(fname, data, ftype='fits'):

    '''
    Write an image to file.

    Args:
        fname (str): output path
        data:        2D numpy array to write
        ftype (str): 'fits' or 'npy'

    '''

    if ftype=='fits':
        with pyfits.PrimaryHDU(data) as hdu:
            hdulist = pyfits.HDUList([hdu])
            hdulist.writeto(fname, clobber=True)
    elif ftype=='npy':
        np.save(fname, data)
    else:
        print('Error: unknown output file type.')
        sys.exit()


def add_padding(dat, width=0.5):

    '''
    Replace NaNs by zeros and add zero padding around an image for the
    following Fourier transforms because the input images are non-periodic.

    Args:
        data:   2D numpy array
        width: padding width (same on all sides) as a fraction of the image size
    Returns:
        padded 2D numpy array with NaNs replaced by zeros
    '''

    lx,ly = data.shape
    f = 2*width+1
    data_pad = np.zeros((int(f*lx), int(f*ly)))
    i0 = int(width*lx)
    j0 = int(width*ly)
    notnan = np.invert(np.isnan(data))
    data_pad[i0:i0+lx, j0:j0+ly][notnan] += data[notnan]
    return data_pad


def data_fft(data):

    '''
    Args:
        dat (2D array): input image
    Returns:
        2D array: normalized Fourier transform of the image.
    '''

    lx,ly = dat.shape
    dataf = fftpack.fftn(data)
    dataf *= np.sqrt(lx*ly)
    vr = np.sqrt((np.abs(data)**2).mean())
    vf = np.sqrt((np.abs(dataf)**2).mean())
    print(f'Checking the Parseval theorem (forward FFT):\nvr={vr}\nvf={vf}')
    return dataf
