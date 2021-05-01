import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal



## read image

img = cv2.imread('natural_scene.jpg',0)
freq_img = np.fft.fft2(img)


### applying gaussian blur to image

# create a 2D-gaussian kernel with the same size of the image
gaussian_kernel = np.outer(signal.gaussian(img.shape[0], 5), signal.gaussian(img.shape[1], 5))
gaussian_freq_kernel = np.fft.fft2(np.fft.ifftshift(gaussian_kernel))

# create a 2D-gaussian kernel with the same size of the image
laplacian_kernel=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])


# enlarge the kernel to the shape of the image

padding = (img.shape[0] - laplacian_kernel.shape[0], img.shape[1] - laplacian_kernel.shape[1])  # total amount of padding
laplacian_kernel = np.pad(laplacian_kernel, (((padding[0]+1)//2, padding[0]//2), ((padding[1]+1)//2, padding[1]//2)), 'constant')
laplacian_freq_kernel = np.fft.fft2(np.fft.ifftshift(laplacian_kernel))

## multiplication in freq domain instead of convolution

filtered_img = laplacian_freq_kernel * gaussian_freq_kernel* freq_img
enhanced_img = np.real (np.fft.ifft2(filtered_img))


# only for plotting
fshift = np.fft.fftshift(freq_img)
magnitude_spectrum_img = 20*np.log(np.abs(fshift))
magnitude_spectrum_filter = 20*np.log(np.abs(laplacian_freq_kernel))

plt.subplot(321),plt.imshow(img, cmap = 'gray')
plt.title('Input Image in grayscale'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(magnitude_spectrum_img, cmap = 'gray')
plt.title('Magnitude Spectrum of image'), plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(magnitude_spectrum_filter,cmap = 'gray')
plt.title('Magnitude Spectrum of LoG'), plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(enhanced_img,cmap = 'gray')
plt.title('Enhanced Image'), plt.xticks([]), plt.yticks([])


# reshape the image to a 2D array of pixels and 1 color value (BW)
pixel_values = enhanced_img.reshape((-1, 1))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 2
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# flatten the center values array
center_values = centers.squeeze()
# flatten the labels array
labels = labels.flatten()

## find cluster with larger mean and its label

text_cluster_center = np.max (center_values)
text_cluster_label = np.where(center_values == center_values.max())

non_text_cluster_center = np.min (center_values)
non_text_cluster_label = np.where(center_values == center_values.min())

### create black image to paint the white cluster pixels on
text_cluster_img = np.zeros([pixel_values.shape[0], pixel_values.shape[1]],dtype=np.uint8).flatten()
non_text_cluster_img = np.zeros([pixel_values.shape[0], pixel_values.shape[1]],dtype=np.uint8).flatten()

text_cluster_img [np.where (labels == text_cluster_label)[1]] = 255
non_text_cluster_img [np.where (labels == non_text_cluster_label)[1]]= 255

### reshape images to display

text_cluster_img = np.reshape(text_cluster_img, enhanced_img.shape)
non_text_cluster_img = np.reshape(non_text_cluster_img, enhanced_img.shape)



### display all images

plt.subplot(325),plt.imshow(text_cluster_img,cmap = 'gray')
plt.title('Text Cluster Image'), plt.xticks([]), plt.yticks([])

plt.subplot(326),plt.imshow(non_text_cluster_img,cmap = 'gray')
plt.title('Non -Text Cluster Image'), plt.xticks([]), plt.yticks([])

plt.show()


