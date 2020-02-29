import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

image=np.mgrid[-5:5, -5:5][0]
image

kernel = np.zeros((5,5))
kernel[1,1]=1
kernel


a='fewf'
b='fewf'
a==b

kh,kw=kernel.shape
ih=image.shape[0]
iw=image.shape[1]

ndim=kernel.ndim
ndim
hpad:int=(kh-1)//2
wpad:int=(kw-1)//2


# y=np.ones((5,5,6))
# x=np.sum(y)
# x

# # i=j=0
# cropped=kernel[0:3,0:3]
# cropped [1,2,0]
# kernel [1,2,0]

# # kernel = np.zeros((5,5,7))
# # cropped=kernel[0:3,0:3,:]
# # print(cropped)
# # cropped.shape
# # image[i:i+kh,j:j+kw].shape
# # kernel.shape
# for paddiing and getting the same dimesnsions we have to pad the dim with (n-1)/2
# where n is the dimension of the filter in that direction
paddedImg=np.pad(image,((hpad,hpad),(wpad,wpad)),mode='reflect')
paddedImg
filtered=np.zeros_like(image)
for i in range(0,ih):
    for j in range(0,iw):
        cropped=paddedImg[i:i+kh,j:j+kw]
        filtered[i,j]=np.sum(np.multiply(kernel,cropped))

filtered-image        