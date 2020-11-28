from skimage import  transform
import numpy as np 
from scipy import ndimage
import random
import cv2
from torchvision import transforms 

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img  = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)
        k = np.max(img.shape)//20*2+1
        bg = cv2.medianBlur(img, k)
        return cv2.addWeighted (img, 4, bg, -4, 128), label

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        #return img, label


class RandomShear(object):
    """ Shear the Image of particular range"""
    def __init__(self, rang, p=0.5):
        assert isinstance( p, (float))
        assert isinstance(rang, (float))
        self.p = p
        self.rang = rang

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        tform = transform.AffineTransform(
        shear = self.rang)
        if np.random.rand(1) < self.p:
            #print("applied shear transformation")
            # Apply transmform to mask if the task is segementation
            tf_img = transform.warp(image, tform.inverse)
            return tf_img, label
        return image, label


class RandomShift(object):
    """ Shear the Image of particular range"""
    def __init__(self, rang, p=0.5):
        assert isinstance( p, (float))
        assert isinstance(rang, (float,tuple))
        self.p = p
        self.rang = rang

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        tform = transform.AffineTransform(
        translation = self.rang)
        if np.random.rand(1) < self.p:
           # print("applied translation transformation")
            # Apply transmform to mask if the task is segementation
            tf_img = transform.warp(image, tform.inverse)
            return tf_img, label
        return image, label

class RandomRotate(object):
    """ Shear the Image of particular range"""
    def __init__(self, rang, p=0.5):
        assert isinstance( p, (float))
        assert isinstance(rang, (float,tuple))
        self.p = p
        self.rang = rang

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        #tform = transform.AffineTransform(
        #rotation = self.rang)
        if np.random.rand(1) < self.p:
           # print("applied rotate transformation")
            # Apply transmform to mask if the task is segementation
            tf_img = transform.rotate(image, self.rang)
            return tf_img, label
        return image, label

class RandomScaling(object):
    """ Shear the Image of particular range"""
    def __init__(self, rang, p=0.5):
        assert isinstance( p, (float))
        assert isinstance(rang, (float,tuple))
        self.p = p
        self.rang = rang

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        tform = transform.AffineTransform(
        scale = self.rang)
        if np.random.rand(1) < self.p:
           # print("applied scale transformation")
            # Apply transmform to mask if the task is segementation
            tf_img = transform.warp(image, tform.inverse)
            return tf_img, label
        return image, label

# class Sharpen(object):
#     def __init__(self, p):
#         self.p = p
        

#     def __call__(self, sample):
        
#         image, label = sample[0], sample[1]
#         t = np.linspace(-10, 10, 30)
#         bump = np.exp(-0.1*t**2)
#         bump /= np.trapz(bump) # normalize the integral to 1

#         # make a 2-D kernel out of it
#         kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#         im_blur = ndimage.convolve(sample, kernel.reshape(30,30,1))

#         im_sharp = np.clip(2*sample - im_blur, 0, 1)
#         if np.random.rand(1) < self.p:
#             print("applied scale transformation")
#             # Apply transmform to mask if the task is segementation
#             tf_img = transform.warp(image, tform.inverse)
#             return tf_img, label
#         return im_sharp, label

class Brightness(object):
    def __init__(self, p):
        self.p = p
        

    def __call__(self, sample):
        img, label = sample[0], sample[1]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - self.p
        v[v > lim] = 255
        v[v <= lim] += self.p

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        

        return img, label

class Sharpen(object):
    def __init__(self, p):
        self.p = p
        

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        #adjusted = cv2.convertScaleAbs(sharpened, alpha=3, beta=20)
        buf = cv2.addWeighted(sharpened, 1.2, sharpened, 0, 0)

        return buf, label

class Contrast(object):
    def __init__(self, p):
        self.p = p
        self.coef = np.array([[[0.299, 0.587, 0.114]]])


    def __call__(self, sample):
        image, label = sample[0], sample[1]
        alpha = 1.0 + random.uniform(-self.p, self.p)
        gray = image * self.coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = (image*alpha)
        image += gray
        print(image.shape)
       
        

        return image, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image /= 255

        image = image*(1.0/255)
        b = np.zeros(5)
        for i in range(0, label+1):
            b[i] = 1
        label = b
        image = image.transpose((2, 0, 1))
        #image = transforms.Normalize(image, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        
        return image, label


        
        

        