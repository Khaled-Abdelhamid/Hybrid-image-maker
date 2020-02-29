import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy.signal import convolve2d, correlate2d
from scipy import fftpack


def my_imfilter(image, kernel,mode ='zeros'):
  """
 This is a fundamental image processing tool (see Chapter 3.2 of Szeliski to learn about image filtering, specifically about linear filtering)
 . Python’s scipy, openCV and scikit-image have efficient functions to perform image filtering, but we will write our own from
 scratch via ​convolution​.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  # get the kernel dimesnsions in kh and kw
  kh,kw=kernel.shape
# get the hight and width only as the third dimension may exist or not
  ih=image.shape[0]
  iw=image.shape[1]
# get the number of dimsensions for both the kernel and the image
  kdim=kernel.ndim
  idim=image.ndim

  assert kdim == 2 , "kernel dimensions must be exaxctly two"
  assert idim == 2 or idim ==3 , "image dimensions must be exaxctly two or three"
  assert (kh%2) !=0 and (kw%2) !=0 , "all kernel dimensions must be odd"
# this foarmula calculates how many rows do i need to put so that i can make proper padding
  hpad=(kh-1)//2
  wpad=(kw-1)//2

  filtered_image = np.zeros_like(image)

  if mode=='zeros':
    md='constant'
  elif mode=='reflect':
    md='reflect'
  else:
    raise   Exception('the mode {} is not defined \n "zeros" and "reflect are available"'.format(x))
# change the padding function according to the input image
  if idim==2:
     paddedImg=np.pad(image,[(hpad,hpad),(wpad,wpad)],mode=md)
  else:
    paddedImg=np.pad(image,[(hpad,hpad),(wpad,wpad),(0,0)],mode='constant')
    dim_No=image.shape[2]
# apply convolution over the number of channels
  for dim in range(0,dim_No):
    for i in range(0,ih):
      for j in range(0,iw):
        # multiply the kernel with the cropped part of the image and then sum the result
          cropped=paddedImg[i:i+kh,j:j+kw,dim]
          filtered_image[i,j,dim]=np.sum(np.multiply(kernel,cropped))

  return filtered_image

def gen_hybrid_image(image1, image2, cutoff_frequency):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  # blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel

  ksize = 15
  sigma = cutoff_frequency


  # Here I do outer product of 2 gaussian vectors to get 2D kernel
  x = np.arange(-ksize//2, ksize//2+1)
  gx = np.exp(-(x)**2/(2*sigma**2))
  g = np.outer(gx, gx)
  g /= np.sum(g)
  kernel = g
  

  
  # Your code here:
  # looping over the channels of the image to apply the gaussian kernel
  low_frequencies = np.zeros(image1.shape, dtype=np.float32)
  
  for i in range(image1.shape[2]):
    low_frequencies[:,:,i] = correlate2d(image1[:,:,i], kernel, 'same') # Replace with your implementation

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  
  low_frequencies2 = np.zeros(image2.shape, dtype=np.float32)
  
  for i in range(image1.shape[2]):
    low_frequencies2[:,:,i] = correlate2d(image2[:,:,i], kernel,'same')





  high_frequencies = image2 - low_frequencies2 # Replace with your implementation


  # print(np.sum(high_frequencies<0))

  # (3) Combine the high frequencies and low frequencies
  
  hybrid_image = low_frequencies/2 + high_frequencies/2 # Replace with your implementation

  high_frequencies = np.clip(high_frequencies,-1.0,1.0)
  hybrid_image = np.clip(hybrid_image,0,1)
  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!


  

  # np.clip(low_frequencies,0,1)
  # np.clip(high_frequencies,0,1)
  # np.clip(hybrid_image,0,1)
  return low_frequencies, high_frequencies, hybrid_image



#######################################################################################################################

# I used the fft_convolve in lab_3 as a helper function to do fft_based convolution on each channel of the image
def fft_convolve(img, filter):
  img_in_freq = fftpack.fft2(img)

  filter_in_freq = fftpack.fft2(filter, img.shape)
  filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
  filtered_img = fftpack.ifft2(filtered_img_in_freq)

  img_in_freq_domain = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
  img_in_freq_domain /= img_in_freq_domain.max() - img_in_freq_domain.min()
  
  filter_in_freq_domain = fftpack.fftshift(np.log(np.abs(filter_in_freq)+1))
  filter_in_freq_domain /= filter_in_freq_domain.max() - filter_in_freq_domain.min()
  
  filtered_img_in_freq_domain = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))
  filtered_img_in_freq_domain /= filtered_img_in_freq_domain.max() - filtered_img_in_freq_domain.min()
  
  filtered_img = np.abs(filtered_img)
  filtered_img = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min())

  return filtered_img






def gen_hybrid_image_fft(image1, image2, cutoff_frequency):
  assert image1.shape == image2.shape

  ksize = 15
  sigma = cutoff_frequency

  x = np.arange(-ksize//2, ksize//2+1)
  gx = np.exp(-(x)**2/(2*sigma**2))
  g = np.outer(gx, gx)
  g /= np.sum(g)
  kernel = g


# Applyingthe fft convolution on each channel
  low_freqs = np.zeros(image1.shape)
  for i in range(image1.shape[2]):
    low_freqs[:,:,i] = fft_convolve(image1[:,:,i], kernel)



  low_freqs2 = np.zeros(image2.shape)
  for i in range(image1.shape[2]):
    low_freqs2[:,:,i] = fft_convolve(image2[:,:,i], kernel)
 

 # getting only high freqs of image2
  high_freqs = image2 - low_freqs2
  

# combining the low freqs and high freqs
  hybrid_image = low_freqs/2 + high_freqs/2 # Replace with your implementation

  high_freqs = np.clip(high_freqs,-1.0,0.5)
  hybrid_image = np.clip(hybrid_image,0,1)

  return low_freqs, high_freqs, hybrid_image



#######################################################################################################################

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect',multichannel = True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
