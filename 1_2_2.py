import os
import math
import numpy as np
from PIL import Image
gaussian_filter = __import__('1_2_1').gaussian_filter 
convolve2d = __import__('1_2_1').convolve2d 
def sobel_filter(image):
    """
    Args:
        image (numpy array): input array

    Return:
        Sobel filtered image of gradient magnitude (numpy array),
        Sobel filtered image of gradient direction (numpy array)
    """

    # TODO: Your code

    #sobel operator for x
    sobel_operator_x = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]])
    #sobel operator for y
    sobel_operator_y = np.array([[1, 2, 1], 
                               [0, 0, 0], 
                               [-1, -2, -1]])
    sobeled_x = convolve2d(image, sobel_operator_x)
    sobeled_y = convolve2d(image, sobel_operator_y)

    #initialize answers
    mag = np.empty((len(image), len(image[0])))
    direction = np.empty((len(image), len(image[0])))

    #calculate mag and direction
    for i in range(len(image)):
        for j in range(len(image[0])):
            gx = sobeled_x[i][j]
            gy = sobeled_y[i][j]
            mag[i][j] = math.sqrt(gx**2 + gy**2)
            direction[i][j] = math.atan2(gy, gx)

    return mag, direction

def non_max_suppression(gradient_magnitude, gradient_orientation):
    """
    Args:
        gradient_magnitude (numpy array): Sobel filtered image of gradient magnitude
        gradient_orientation (numpy array): Sobel filtered image of gradient direction

    Return:
        Non-maximum suppressed image (numpy array)
    """

    # TODO: Your code
    image = np.empty((len(gradient_magnitude), len(gradient_magnitude[0])))

    #for each pixel
    for i in range(len(gradient_magnitude)):
        for j in range(len(gradient_magnitude[0])):
            cur_mag = gradient_magnitude[i, j]
            cur_ori = gradient_orientation[i, j]

            #edge case
            if cur_ori < 0:
                cur_ori += 180
            #anaylze the gradient direction for current pixel
            if (0 <= cur_ori < 22.5) or (157.5 <= cur_ori <= 180):
                a = i
                b = j + 1
                c = i
                d = j - 1
            elif 22.5 <= cur_ori < 67.5:
                a = i - 1
                b = j + 1
                c = i + 1
                d = j - 1
            elif 67.5 <= cur_ori < 112.5:
                a = i - 1
                b = j
                c = i + 1
                d = j
            else:  
                a = i - 1
                b = j - 1
                c = i + 1
                d = j + 1

            #define our neighbors p & r
            try:
                pixel_p = gradient_magnitude[a][b]
            except IndexError:
                pixel_p = 0
            try:
                pixel_r = gradient_magnitude[c][d]
            except IndexError:
                pixel_r = 0
                
            #if current pixel is the maximum
            if cur_mag >= pixel_p and cur_mag >= pixel_r:
                image[i, j] = cur_mag
            else:
                image[i, j] = 0
    return image

def double_threshold(image, low_threshold_rate, high_threshold_rate):
    """
    Args:
        image (numpy array): Non-maximum suppressed image
        low_threshold_rate (float): used to identify non-relevant pixels
        high_threshold_rate (float): used to identify strong pixels

    Return:
        Double-thresholded image (numpy array)
    """

    # TODO: Your code
    doubled = np.empty((len(image), len(image[0])))

    #Get the high and low threshold
    high_threshold = np.max(image) * high_threshold_rate
    low_threshold = high_threshold * low_threshold_rate

    #define weak and strong variables
    weak = np.int32(25)
    strong = np.int32(255)

    #for each pixel
    for i in range(len(image)):
        for j in range(len(image[0])):
            cur_pixel = image[i][j]

            #Higher thresholds assigned strong value,
            #In between high and low are assigned weak
            #irrelevant are assigned 0
            if cur_pixel >= high_threshold:
                doubled[i][j] = strong
            elif low_threshold <= cur_pixel < high_threshold:
                doubled[i][j] = weak
            else:
                doubled[i][j] = 0
    return doubled

def hysteresis_threshold(image):
    """
    Args:
        image (numpy array): Double-thresholded image

    Return:
        Hysteresis-thresholded image (numpy array)
    """

    # TODO: Your code
    hysteresised = np.empty((len(image), len(image[0])))

    #weak and strong values
    weak = np.int32(25)
    strong = np.int32(255)

    #for ech pixel
    for i in range(len(image)):
        for j in range(len(image[0])):
            cur_pixel = image[i][j]
            #if pixel is weak then see if there are neighboring strong pixels
            #if so, then change the weak pixel to a strong
            if cur_pixel == weak:
                try:
                    if ((image[i+1][j-1] == strong) or (image[i+1][j] == strong) or (image[i+1][j+1] == strong)
                        or (image[i][j-1] == strong) or (image[i][j+1] == strong)
                        or (image[i-1][j-1] == strong) or (image[i-1][j] == strong) or (image[i-1][j+1] == strong)):
                        hysteresised[i][j] = strong
                    else:
                        hysteresised[i][j] = 0 
                except IndexError:
                    pass
            elif image[i][j] == strong:
                hysteresised[i][j] = strong
            
    return hysteresised


if __name__ == "__main__":
    sigma = 1.5
    kernel_size = 2
    low_threshold_rate = 0.05
    high_threshold_rate = 0.15

    img = np.asarray(Image.open('xyz.jpg').convert('L'))
    img = img.astype('float32')

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    gaussian_filtered = gaussian_filter(img, sigma)

    grad_magnitude, grad_direction = sobel_filter(gaussian_filtered)
    if grad_magnitude is not None and grad_direction is not None:
        Image.fromarray(grad_magnitude.astype('uint8')).save(os.path.join(logdir, 'sobel_filter_grad_magnitude.jpeg'))
        Image.fromarray(grad_magnitude.astype('uint8')).show()
        Image.fromarray(grad_direction.astype('uint8')).save(os.path.join(logdir, 'sobel_filter_grad_direction.jpeg'))
        Image.fromarray(grad_direction.astype('uint8')).show()

    ret = non_max_suppression(grad_magnitude, grad_direction)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'non_max_suppression.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()

    ret = double_threshold(ret, low_threshold_rate, high_threshold_rate)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'double_threshold.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()

    ret = hysteresis_threshold(ret)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'hysteresis_threshold.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()
