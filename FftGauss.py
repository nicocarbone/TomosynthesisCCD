import numpy as np
from scipy import ndimage

def fft_gauss(in_array, kernel_size):
    '''
    Perform FFR Gauss low-pass filtering
    
    in_array: input image
    
    kernel_size: kernel size for smoothing
    '''
    in_array[np.isinf(in_array)] = 0
    in_array[np.isnan(in_array)] = 0 

    # FFT filter
    im_fft = np.fft.rfftn(in_array)
    im_rfft_filtered = ndimage.fourier_gaussian(im_fft, kernel_size, 
                                                in_array.shape[1])
    im_filtered = np.fft.irfftn(im_rfft_filtered)
    
    # Power spectrum       
    pwr_spectrum = abs(np.fft.fftshift(im_fft))**2
    pwr_spectrum_filtered = abs(np.fft.fftshift(im_rfft_filtered))**2
        
    # Re-normalization
    sum_ratio = in_array.sum()/im_filtered.sum()
    im_filtered = im_filtered*sum_ratio
    
    return im_filtered, pwr_spectrum, pwr_spectrum_filtered 