import os
import numpy as np
from PIL import Image

def compute_fourier_transform(image):
    """
    Args:
        image (numpy array): input array

    Return:
        f_shift (numpy array): The shifted Fourier Transform with the zero-frequency component at the center.
        magnitude_spectrum (numpy array): The logarithmic magnitude spectrum of the transformed image.
    """
    # TODO: Your code

    f_shift = np.fft.fftshift(np.fft.fft2(image))

    return f_shift, np.log(1 + np.abs(f_shift))

def create_low_pass_filter(shape, cutoff):
    """
    Args:
        shape (tuple): The dimensions of the filter (same as the image dimensions).
                     Should be in the form (rows, cols).
        cutoff (int): The cutoff frequency for the low-pass filter. 
                    Determines how much of the low frequencies are retained.
                    
    Return:
        mask (numpy array): A binary mask with ones in the low-frequency region and 
                      zeros in the high-frequency region.
    """
    
    # TODO: Your code
    mask = np.empty((len(shape), len(shape[0])))
    midX, midY = len(shape) // 2, len(shape[0]) // 2  

    for i in range(len(shape)):
        for j in range(len(shape[0])):
            #Euclidian formula
            distance = np.sqrt((i - midX) ** 2 + (j - midY) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1  
            else:
                mask[i, j] = 0  

    return mask

def create_high_pass_filter(shape, cutoff):
    """
    Args:
        shape (tuple): The dimensions of the filter (same as the image dimensions).
                     Should be in the form (rows, cols).
        cutoff (int): The cutoff frequency for the high-pass filter.
                    Determines how much of the high frequencies are retained.
                    
    Returns:
        mask (numpy array): A binary mask with ones in the high-frequency region and 
                      zeros in the low-frequency region.
    """
    # TODO: Your code
    mask = np.empty((len(shape), len(shape[0])))
    midX, midY = len(shape) // 2, len(shape[0]) // 2  

    for i in range(len(shape)):
        for j in range(len(shape[0])):
            #Euclidian formula
            distance = np.sqrt((i - midX) ** 2 + (j - midY) ** 2)
            if distance > cutoff:
                mask[i, j] = 1  
            else:
                mask[i, j] = 0  

    return mask


if __name__ == "__main__":

    image = np.asarray(Image.open('xyz.jpg').convert('L'))
    image = image.astype('float32')
    cutoff = 30  # Example cutoff frequency
    f_shift, magnitude_spectrum = compute_fourier_transform(image)


    # Low-pass filter
    low_pass_filter = create_low_pass_filter(image.shape, cutoff)
    f_low_pass = f_shift * low_pass_filter
    low_pass_magnitude_spectrum = np.log(1 + np.abs(f_low_pass))

    # High-pass filter
    high_pass_filter = create_high_pass_filter(image.shape, cutoff)
    f_high_pass = f_shift * high_pass_filter
    high_pass_magnitude_spectrum = np.log(1 + np.abs(f_high_pass))
