#!/home/chad/anaconda3/envs/tfgpu/bin/python

import sys,os

import numpy as np

import skimage.io as skio
import glob
import re
from skimage.color import rgb2hsv
import time

class data:
    # set the path, image type and whether to convert to hsv
    def __init__(self,path,imType,rgb2hsv=False):
        self.images = np.array(glob.glob(path+'/*.'+imType))
        self.imType = imType
        self.orderImages()
        self.rgb2hsv = rgb2hsv  # True -> convert images to single channel Hue, False -> preserve images as is

    #   this finds the image number of each image and puts them into numeric order instead of alphabetic order
    def orderImages(self):
        # get the numbers in each path name
        z = np.array([list(map(int,re.findall(r'[0-9]+',i))) for i in self.images]) 
        # get the length of string needed for last number in pathname
        zw = np.hstack([np.power(10,np.ceil(np.log10(np.max(z,axis=0))))[1:],[1]]) 
        mi = np.argsort([np.int(np.dot(num,zw)) for num in z])
        self.images = self.images[mi] # re-orders the array
        if self.imType=='npy':
            #self.imNumbers = z[mi,1] # usefull for numpy arrays to determine how many to read in
            self.imNumbers = z[mi,0] # usefull for numpy arrays to determine how many to read in

    # these are assumed to be in the correct image format RGB, HSV, H etc
    def getNumpy(self,num):
        mask = self.imNumbers<num+self.imNumbers[0] ## start from the first index in the sequence
        images = self.images[mask]
        immat = np.concatenate([np.load(image) for image in images],axis=0)[:num]

        numSamples = immat.shape[0]
        return immat,numSamples

    def getImages(self,num):
        images = self.images[:num]
        if self.rgb2hsv:
            immat = np.stack([rgb2hsv(skio.imread(image))[:,:,0] for image in images],axis=0)
        else:
            immat = np.stack([skio.imread(image) for image in images],axis=0)/255.0
        numSamples = immat.shape[0]
        return immat, numSamples

    #def makeInteriorSequences(self,numSequences,frameLenIn,frameLenOut,validationSet=1):
    def makeInteriorSequences(self,numSequences, input_skip,input_len,validationSet=1):
        num = input_skip*(input_len-1)+input_skip*(numSequences+validationSet)
#        print(num)
        if self.imType == 'npy':
            immat, numSamples = self.getNumpy(num)
        else:
            immat, numSamples = self.getImages(num)

#        print('immat.shape',immat.shape)
        input_num = int(numSamples/(input_skip))-input_len+1
        input_center = int(input_len/2)

        imageSequencesInputTrain = np.stack([immat[::input_skip,...][i:i+input_len,...] for i in range(input_num)])
        imageSequencesOutputTrain = np.stack([immat[input_skip*(i+input_center-1)+1:input_skip*(i+input_center),...] for i in range(input_num)])
#        print('imageSequencesOutputTrain=',self.imageSequencesOutputTrain.shape)

        if imageSequencesInputTrain.shape[0]<num:
            ntrain = int(imageSequencesInputTrain.shape[0]*numSequences/(numSequences+validationSet))
            nvalid = int(imageSequencesInputTrain.shape[0]*validationSet/(numSequences+validationSet))

            numSequences = ntrain
            validationSet = nvalid
        
        ### TODO make the validation data separate from the training data
        self.imageSequencesInputTrain = imageSequencesInputTrain[:numSequences]
        self.imageSequencesOutputTrain = imageSequencesOutputTrain[:numSequences]

        self.imageSequencesInputValid = imageSequencesInputTrain[numSequences:]
        self.imageSequencesOutputValid = imageSequencesOutputTrain[numSequences:]

        self.inputSize  = list(self.imageSequencesInputTrain.shape[-4:])
        self.outputSize = list(self.imageSequencesOutputTrain.shape[-4:])

        print('*'*88)
        print(self.inputSize,self.outputSize)
        print(self.imageSequencesInputTrain.shape)
        print(self.imageSequencesOutputTrain.shape)

        print(self.imageSequencesInputValid.shape)
        print(self.imageSequencesOutputValid.shape)

    def makeSequences(self,numSequences,frameLenIn,frameLenOut,validationSet=1):
        num = numSequences+frameLenIn+frameLenOut-1+validationSet
        if self.imType == 'npy':
            immat, numSamples = self.getNumpy(num)
        else:
            immat, numSamples = self.getImages(num)

        # The following lines of code generate input/ouput pairs for the ANN to train on.
        #   these pairs will be chronological sequences of frames.  For example a single input/output pair would look like:
        #   input = [Im_i, Im_i+1, Im_i+2, Im_i+3]
        #   output = [Im_i+4, Im_i+5]
        #   where in this case frameLenIn=4, and frameLenOut=2
        #   where Im_j is an n dimensional matrix representing a single image
        #   The following lines will generate two matrices where the first index will return one of the input or output from above
        
        # these lines of code take a matrix of images of dim [numImageSamples, H,W,C]
        #   and convert it into sequences of images of dim [numSequenceSamples,frameLenIn,H,W,C]
        #   for example (in lower dimensions) if we have a sequence of images: [Im_0, Im_1, Im_2, ...]
        #   we will get a new sequence that looks like : [[Im_0, Im_1, Im_2],[Im_1, Im_2, Im_3], ...]
        #   where each of the Im_* is a [H,W,C] matrix
        zz = [immat[i:numSamples-frameLenIn-frameLenOut+1+i] for i in range(frameLenIn+frameLenOut)]
        print('len zz = ',len(zz), 'zz0.shape = ',zz[0].shape)
        zz = np.stack(zz,axis=1)
        print('zz.shape = ',zz.shape)
        self.imageSequencesInputTrain = zz[:-validationSet,:frameLenIn]
        self.imageSequencesOutputTrain = zz[:-validationSet,frameLenIn:]

        self.imageSequencesInputValid = zz[-validationSet:,:frameLenIn]
        self.imageSequencesOutputValid = zz[-validationSet:,frameLenIn:]

        if len(self.imageSequencesInputTrain.shape)==4:
            self.imageSequencesOutputTrain = np.expand_dims(self.imageSequencesOutputTrain,axis=-1)
            self.imageSequencesOutputValid = np.expand_dims(self.imageSequencesOutputValid,axis=-1)

            self.imageSequencesInputTrain = np.expand_dims(self.imageSequencesInputTrain,axis=-1)
            self.imageSequencesInputValid = np.expand_dims(self.imageSequencesInputValid,axis=-1)

        self.inputSize  = list(self.imageSequencesInputTrain.shape[-4:])
        self.outputSize = list(self.imageSequencesOutputTrain.shape[-4:])

    def makeOffsetSequences(self,numSequences,frameLenIn,validationSet=1):
        frameLenOut = 1 #frameLenIn
        num = numSequences+frameLenIn+frameLenOut-1+validationSet
        if self.imType == 'npy':
            immat, numSamples = self.getNumpy(num)
        else:
            immat, numSamples = self.getImages(num)

        # The following lines of code generate input/ouput pairs for the ANN to train on.
        #   these pairs will be chronological sequences of frames.  For example a single input/output pair would look like:
        #   input =  [Im_i,   Im_i+1, Im_i+2, Im_i+3, ...]
        #   output = [Im_i+1, Im_i+2, Im_i+3, Im_i+4, ...]
        #   where in this case frameLenIn=4, and frameLenOut=2
        #   where Im_j is an n dimensional matrix representing a single image
        #   The following lines will generate two matrices where the first index will return one of the input or output from above
        
        # these lines of code take a matrix of images of dim [numImageSamples, Height,Width,Colors]
        #   and convert it into sequences of images of dim [numSequenceSamples,frameLenIn,H,W,C]
        #   for example (in lower dimensions) if we have a sequence of images: [Im_0, Im_1, Im_2, ...]
        #   we will get a new sequence that looks like : [[Im_0, Im_1, Im_2],[Im_1, Im_2, Im_3], ...]
        #   where each of the Im_* is a [H,W,C] matrix
        zz = [immat[i:numSamples-frameLenIn-frameLenOut+1+i] for i in range(frameLenIn+frameLenOut)]
        print('len zz = ',len(zz), 'zz0.shape = ',zz[0].shape)
        zz = np.stack(zz,axis=1)
        print('zz.shape = ',zz.shape)
        self.imageSequencesInputTrain = zz[:-validationSet,:-1]
        self.imageSequencesOutputTrain = zz[:-validationSet,1:]

        self.imageSequencesInputValid = zz[-validationSet:,:-1]
        self.imageSequencesOutputValid = zz[-validationSet:,1:]

        if len(self.imageSequencesInputTrain.shape)==4:
            self.imageSequencesOutputTrain = np.expand_dims(self.imageSequencesOutputTrain,axis=-1)
            self.imageSequencesOutputValid = np.expand_dims(self.imageSequencesOutputValid,axis=-1)

            self.imageSequencesInputTrain = np.expand_dims(self.imageSequencesInputTrain,axis=-1)
            self.imageSequencesInputValid = np.expand_dims(self.imageSequencesInputValid,axis=-1)

        self.inputSize  = list(self.imageSequencesInputTrain.shape[-4:])
        self.outputSize = list(self.imageSequencesOutputTrain.shape[-4:])

    def makeSingles(self,numSingles, validationSet=1):
        num = numSingles+validationSet
        if self.imType == 'npy':
            immat, numSamples = self.getNumpy(num)
        else:
            immat, numSamples = self.getImages(num)

        self.imageSequencesInputTrain = immat[:numSingles]
        self.imageSequencesOutputTrain = immat[:numSingles]

        self.imageSequencesInputValid = immat[numSingles:]
        self.imageSequencesOutputValid = immat[numSingles:]

        self.inputSize  = list(self.imageSequencesInputTrain.shape[1:])
        self.outputSize = list(self.imageSequencesOutputTrain.shape[1:])

    def getInputSize(self):
        return self.inputSize

    def getOutputSize(self):
        return self.outputSize

    def getTrainingImages(self):
        return self.imageSequencesInputTrain, self.imageSequencesOutputTrain

    def getValidationImages(self):
        return self.imageSequencesInputValid, self.imageSequencesOutputValid


if __name__=='__main__':
    x = data('/home/chad/data/olcmen/cfd','npy',False)
    x.makeInteriorSequences(7, 5,6)
