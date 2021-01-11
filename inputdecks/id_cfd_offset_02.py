import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
import importlib

# data 
class params:
    def __init__(self):
        self.caseName = self.__module__ # caseName is the name of thid file without the extension
        self.getData()

        ############################################################################
        # custom loss functions
        def mimse(y,yh):
            factor = 2.0
            axis = range(1,len(y.shape))
            err = tf.math.reduce_mean(tf.math.squared_difference(y,yh),axis=axis)
            return err

        def ssimloss(y,yh):
            ret = tf.image.ssim(y,yh,1.0)
            return ret

        def lastloss(y,yh):
            ret = tf.math.reduce_mean(tf.math.squared_difference(y[:,0],yh[:,0]))
            #ret = tf.norm(yh[0])+tf.math.reduce_mean(tf.math.squared_difference(y[1],yh[1]))
            #ret = tf.math.reduce_mean(tf.math.squared_difference(y[-1],yh[-1]))
            #print(y.shape,yh.shape)
            return ret

        self.customLossFunction = mimse
        self.customLossFunction = ssimloss
        self.customLossFunction = lastloss
        self.run = False
        self.run = True

        self.training()

    def getData(self):
        self.dataPath = "/home/chad/data/olcmen/cfd/"
        self.dataExt = "jpeg"
        self.dataExt = "npy"
        self.datargb2hsv = True
        self.dataTrainSamples = 100
        self.dataNumIn = 5
        self.dataNumOut = 4
        self.dataValidSamples = 10
        self.input_skip = 5
        self.input_len = 6
        self.predictType = {'type':'offset','numSequences':1000, 'frameLenIn':5,'numValidation':100}
        #self.predictType = {'type':'next','numSequences':1000, 'frameLenIn':5, 'frameLenOut':1,'validationSet':1}
        #self.predictType = {'type':'center','numSequences':1000, 'input_skip':5, 'input_len':6,'validationSet':1}


    def getModel(self,inputSize,outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.model()

    def model(self):
        activation = 'sigmoid'
        self.annLayers = [
            keras.Input(shape=self.inputSize),
        
            layers.ConvLSTM2D(filters=10, kernel_size=(10,10), strides=1,padding="valid", return_sequences=True, activation=activation,name='clstm_0'),
            layers.ConvLSTM2D(filters=10, kernel_size=(4, 4),  strides=1,padding="valid", return_sequences=True, activation=activation,name='clstm_1'),
            #layers.ConvLSTM2D(filters=10, kernel_size=(4, 4), strides=1,padding="valid", return_sequences=True, activation=activation,name='clstm_2'),

            #layers.Conv3DTranspose(filters=10,kernel_size=(1,4,4),activation=activation),
            layers.Conv3DTranspose(filters=10,kernel_size=(1,4,4),activation=activation),
            layers.Conv3DTranspose(filters=1,kernel_size=(1,10,10),activation=activation),
        
            #layers.Permute((3,1,2,4))
        ]

        # fit parameters
        self.annOpt = 'Adam'
        self.annLoss = self.customLossFunction
        self.annLoss = 'MSE'

    def training(self):
        # training parameters
        self.modelBatchSize = 1
        self.modelEpochs = 10
        self.modelValidSplit = 0.1
