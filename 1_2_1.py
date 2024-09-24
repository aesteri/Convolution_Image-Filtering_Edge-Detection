import os
import sys
import math
import numpy as np
from PIL import Image
# import scipy 

def convolve2d(image, kernel):
    """
    Args:
        image (numpy array): input array
        kernel (numpy array): kernel to be applied

    Return:
        convolved image (numpy array)
    """
    # TODO: Your code
    
    # initialize the output array with og dimensions 
    answer = np.empty((len(image), len(image[0])))
    # pad the image
    padded = np.pad(image, pad_width=((len(kernel)//2, len(kernel)//2), (len(kernel[0])//2, len(kernel[0])//2)), mode='reflect')
    # flip the kernel
    flipped = np.flip(kernel)
    for i in range(len(image)): 
        for j in range(len(image[0])):
            total = 0
            # iterate over the padded image KERNAL times
            for a in range(len(flipped)):
                for b in range(len(flipped[0])):
                    # add the product of the corresponding selection and the flipped kernel
                    #  and add it to the sum
                    total += padded[i + a][j + b] * flipped[a][b]
            answer[i][j] = total
    return answer



def laplacian_filter(image, laplacian_kernel):
    """
    Args:
        image (numpy array): input array
        laplacian_kernel (numpy array): kernel to be applied

    Return:
        convolved image (numpy array)
    """
    # TODO: Your code
    return convolve2d(image, laplacian_kernel)

    

def gaussian_filter(image, sigma: float) -> np.ndarray:
    """
    Args:
        image (numpy array): input array
        sigma (float): standard deviation of gaussian filter

    Return:
        convolved image (numpy array)
    """
    kernel_size = math.ceil(sigma * 6)
    # TODO: Your code
    kernel = np.empty((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            #use the gaussian formula
            first_part = 1 / (2.0 * np.pi * sigma**2)
            second_part = np.exp(-((i**2 + j**2) / (2.0*sigma**2))) * first_part

            kernel[i][j] = second_part
    #normalize the kernel
    normalized = kernel /np.sum(kernel) 
    
    return convolve2d(image, normalized)


if __name__ == "__main__":
   
    # Do not modify here
    image = np.asarray(Image.open('xyz.jpg').convert('L'))
    image = image.astype('float32')
    laplacian_kernel = np.array([[0.08,  0.12, 0.08],
                                [0.12, 0.20, 0.12],
                                [0.08, 0.12, 0.08]])
    # Laplacian filter test
    student_laplacian = laplacian_filter(image, laplacian_kernel)
    #correct_laplacian = scipy.ndimage.convolve(image, laplacian_kernel)
    Image.fromarray(np.uint8(student_laplacian * 255)).save(f'output_laplacian.jpg')
    #Image.fromarray(np.uint8(correct_laplacian * 255)).save(f'11output_laplacian.jpg')
    # Gaussian filter test
    student_gaussian = gaussian_filter(image, sigma=1.0)
    #correct_gaussian = scipy.ndimage.gaussian_filter(image, sigma=1.0)
    Image.fromarray(np.uint8(student_gaussian * 255)).save(f'output_gaussian.jpg')
    #Image.fromarray(np.uint8(correct_gaussian * 255)).save(f'11output_gaussian.jpg')
