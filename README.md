# 🎠 Computer Vision - Convolution, Edge Detecting, Image Filtering 

------------------------------------------
⭐implementing Computer Vision concepts in python with numPy

⭐1 cup of coffee

## Testing
Get a JPG image in the same directory and rename it to xyz.jpg


## 1_2_1 Implementing Convolution

1. convolve2d(image, kernel)
Implement a function that performs 2D convolution on a grayscale image. The function should handle the padding of the image appropriately (e.g., zero padding) and apply the kernel to the entire image.

2. laplacian filter(image, laplacian kernel)
Implement the laplacian filter for image fin and filter g ∈ R(2k+1)×(2k+1) (provided in in the python file):

3. gaussian filter(image, sigma)
Implement Gaussian smoothing using filter g ∈ R(2k+1)×(2k+1) with standard devia- tion σ.

## 1_2_2 Implementing a Canny Edge Detector

1. sobel filter(image)
Return edge gradient magnitude and edge gradient direction using a sobel filter.

2. non max suppression(gradient magnitude,gradient orientation)
Return non-maximum suppressed image whose edges are a single pixel wide with the given edge gradient magnitude and edge gradient direction images using Non- maximum suppression.

3. double threshold(image,low threshold rate,high threshold rate)
Return double-thresholded image using two given thresholds to identify strong, weak and non-relevant pixels.

4. hysteresis threshold(image)
Return hysteresis-thresholded image, where weak pixels become strong only if there is at least one strong pixel among their neighboring pixels.

## 1_2_3 Frequency Domain Analysis of Image Filtering

1. compute_fourier_transform(image)
Compute the fourier transform of the image. Shift the zero-frequency component to the center. Also, return the magnitude spectrum of the Fourier-transformed image.

2. create_low_pass_filter(shape, cutoff)
Implement a low-pass filter by creating a mask that allows low-frequency components to pass through while blocking high-frequency components. The mask should be designed based on a cutoff frequency.

3. create_high_pass_filter(shape, cutoff)
Implement a high-pass filter by creating a mask that blocks low-frequency components and allows high-frequency components to pass through.


##  Libraries:
* numpy
