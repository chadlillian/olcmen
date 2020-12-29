#!/home/chad/anaconda3/envs/tfgpu/bin/python

import sys,os
import numpy as np
import tensorflow.keras as tfk

#####################################################
# puts the numpy arrays into a data generator
# this can be passed into the fit function of a keras.models.Model
class data(tfk.utils.Sequence):
    def __init__(self):
        None

    def load(self,inputData,outputData):
        self.inputData = inputData
        self.outputData = outputData
        self.dataLength = self.inputData.shape[0]

    def setBatchSize(self,batchsize):
        self.batchSize = batchsize
        self.numBatches = int(np.floor(self.dataLength/self.batchSize))

    def __len__(self):
        return self.numBatches

    def __getitem__(self,index):
        index1 = index*self.batchSize
        index2 = (index+1)*self.batchSize
        self.index = index

        return (self.inputData[index1:index2],self.outputData[index1:index2])
    
    def on_epoch_end(self):
        print(self.index)


if __name__ == "__main__":
    a = data()
    b = np.random.rand(1000,10,5)
    c = np.random.rand(1000,3,2)
    a.load(b,c)
    a.setBatchSize(100)

    for aa in a:
        print(len(aa),aa[0].shape,aa[1].shape)
