#!/home/chad/anaconda3/envs/tfgpu/bin/python

import setproctitle as spt
import sys,os,shutil,glob
spt.setproctitle(sys.argv[0])

import importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, models

#import memory
import imageData as iD
import dataGenerator as dG
import skimage.io as skio

class convlstm:
    def __init__(self,deck):
        self.deck = deck

    def buildModels(self):
        self.loadData()
        self.buildModel()

    def buildModel(self):
        self.deck.getModel(self.inputSize,self.outputSize)
        csv_logger = tf.keras.callbacks.CSVLogger(self.deck.caseName+'/training.log')
        hloutputs = []
        inputs = self.deck.annLayers[0]
        x = inputs
        for hl in self.deck.annLayers[1:-1]:
            x = hl(x)
            if 'output' in hl.name:
                hloutputs.append(x)
        outputs = self.deck.annLayers[-1](x)
        
        self.hiddenLayerOutputs = hloutputs
        self.model = Model(inputs=inputs,outputs=outputs)

        self.model.summary()
        self.model.compile(optimizer=self.deck.annOpt, loss=self.deck.annLoss)

    def loadData(self):
        imdata = iD.data(self.deck.dataPath,self.deck.dataExt,self.deck.datargb2hsv)

        pt = self.deck.predictType
        if pt['type']=='center':
            imdata.makeInteriorSequences(pt['numSequences'],pt['input_skip'],pt['input_len'],validationSet=pt['numValidation'])
        elif pt['type']=='next':
            imdata.makeSequences(pt['numSequences'],pt['frameLenIn'],pt['frameLenOut'],validationSet=pt['numValidation'])
        elif pt['type']=='offset':
            imdata.makeOffsetSequences(pt['numSequences'],pt['frameLenIn'],validationSet=pt['numValidation'])

        print('done loading data')
        print('+'*88)
        imageSequencesInput, imageSequencesOutput = imdata.getTrainingImages()
        imageSequencesInputV, imageSequencesOutputV = imdata.getValidationImages()
        
        inputSize = imdata.getInputSize()
        outputSize = imdata.getOutputSize()

        outputDir = self.deck.caseName
        
        if not os.path.exists(self.deck.caseName):
            os.mkdir(self.deck.caseName)
        shutil.copy(self.deck.caseName+'.py',self.deck.caseName+'/'+self.deck.caseName+'.py')

        np.save(self.deck.caseName+'/valid_in',imageSequencesInputV)
        np.save(self.deck.caseName+'/valid_out',imageSequencesOutputV)
        
        dataGen = dG.data()
        dataGen.load(imageSequencesInput,imageSequencesOutput)
        dataGen.setBatchSize(self.deck.modelBatchSize)

        self.dataGen = dataGen
        self.inputSize = inputSize
        self.outputSize = outputSize

    def train(self):
        self.model.fit(x=self.dataGen,epochs=self.deck.modelEpochs)#,callbacks=[csv_logger])

    def saveModel(self):
        self.model.save(self.deck.caseName+'/model.h5')
        for hlo in self.hiddenLayerOutputs:
            hloname = hlo.name.split('/')[0]
            hlon = hlo.name.split('/')[0]
            modelhlo = Model(inputs=self.model.input,outputs= self.model.get_layer(hlon).output)
            modelhlo.save(self.deck.caseName+'/model_%s.h5'%hloname)

    def loadModels(self):
        outputDir = self.deck.caseName
        modelfiles = glob.glob(outputDir+'/model*.h5')
        self.models = {}
        
        for modelfile in modelfiles:
            mfn = modelfile.split('/')[1].split('.')[0]
            self.models[mfn] = models.load_model(modelfile,compile=False)

    def evalModel(self):
        outputDir = self.deck.caseName
        yh = self._evalModel('model')
        y = np.load(outputDir+'/valid_out.npy')
        num = np.linalg.norm(y-yh,axis=(1,2))
        den = np.linalg.norm(y,axis=(1,2))
        print(np.mean(num/den))

        self.y = y
        self.yh = yh

    def _evalModel(self,modelname):
        outputDir = self.deck.caseName
        x = np.load(outputDir+'/valid_in.npy')

        # predict on batches of same size as trained on
        #  to ensure that the gpu can handle the array size
        n = x.shape[0]//self.deck.modelBatchSize
        xbatches = np.array_split(x,n)
        ybatches = []
        for xb in xbatches:
            yhb = self.models[modelname].predict(xb)
            ybatches.append(yhb)
        yh = np.concatenate(ybatches,axis=0)
        np.save(outputDir+'/valid_out_hat.npy',yh)
        return yh

#   Deprecated, I think
#    def predictModel(self):
#        outputDir = self.deck.caseName
#        x = np.load(outputDir+'/valid_in.npy')
#        nfi = x.shape[1]
#        y = np.load(outputDir+'/valid_out.npy')
#        xyh = np.zeros(x.shape)
#        xyh[0] = x[0].copy()
#
#        for i in range(xyh.shape[0]-1):
#            xi = xyh[None,i]
#            yh = self.models['model'].predict(xi)
#
#            xyh[i+1,:-1] = xi[0,1:].copy()
#            xyh[i+1,-1] = yh[0,0].copy()
#
#        self.yh = xyh[:,-1]
#        self.y = y

    def evalModels(self):
        for modelname in self.models.keys():
            yh = self._evalModel(modelname)
            print(modelname,yh.shape)

    # make images with real on the left, predicted on the right
    def makeImages(self,name='frame_'):
        imagesDir = self.deck.caseName+'/images'
        if not os.path.exists(imagesDir):
            os.mkdir(imagesDir)
        
        #rows = self.y.shape[1]
        for i in range(self.yh.shape[0]):
            z = np.concatenate((self.y[i].squeeze(),self.yh[i].squeeze()),axis=1)
            #print('*'*88)
            #print('*'*88)
            #print('*'*88)
            #print(self.yh.shape)
            #for ii in range(4):
            #    for jj in range(ii):
            #        print(ii,jj,np.linalg.norm(self.yh[i,ii]-self.yh[i][jj]))
            #print('*'*88)
            #print('*'*88)
            #print('*'*88)
            zs = z.shape

            print(self.y[0].shape)
            print(np.linalg.norm(self.y[0]-self.y[1]),np.linalg.norm(self.yh[0]-self.yh[1]))
            zz = np.concatenate([zi for zi in z],axis=1)
            print(self.y.shape,self.yh.shape,z.shape,zz.shape)
            skio.imsave('%s/%s%03i.jpg'%(imagesDir,name,i),zz)
            #skio.imsave('%s/%s%03i.jpg'%(imagesDir,name,i),z[0])

