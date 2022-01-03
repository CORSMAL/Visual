# A class for extracting the dataset

import os
import torch
from torchvision import transforms

import glob
import numpy as np
from PIL import Image
import random
import cv2
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class DataExtractor(object):

    def __init__(self, annotationsHolder, pathImagesFile, configHolder, trainingMinsAndMaxs=None):

        #######################################################################
        # Define parameters
        self.number_of_cameras = 1  # <---------------------------------------------------- For now used one camera only

        # Number of singled valued parameters
        self.numberOfInputSingleValues = 3 * self.number_of_cameras
        self.numberOfOutputSingleValues = configHolder.config['number_of_neurons_final_FC']
        self.image_channels = 3 * self.number_of_cameras

        self.batch_size = configHolder.config['batch_size']
        # Size of images
        self.x_size = configHolder.config['x_size']
        self.y_size = configHolder.config['y_size']

        ######################################################################

        if trainingMinsAndMaxs != None:
            # self.minAverageDistance, self.maxAverageDistance, self.minWidthTop, self.maxWidthTop, \
            # self.minWidthBottom, self.maxWidthBottom = trainingMinsAndMaxs
            self.minAverageDistance, self.maxAverageDistance, self.minMass, self.maxMass = trainingMinsAndMaxs

        self.FindNumberOfImages(pathImagesFile)
        self.FindNumberOfBatches()

        #######################################################################
        # Extraction of the data

        # Extract the images
        inputImages = self.ExtractImagesFromPath(pathImagesFile)

        # Extact the normalization values for the single input elements
        if trainingMinsAndMaxs == None:
            self.ExtractMinAndMaxValuesOfInputsAndOutputs(annotationsHolder)
        else:
            self.CompactMaxsAndMinsValuesOutputs()

        # Extract the single input elements (i.e., annotation inputs and outputs)
        inputSingleValues, outputSingleValues = self.ExtractAnnotationInputsAndOutputs(
            annotationsHolder)
        # Shuffle the data
        inputImagesShuffle, inputSingleValuesShuffle, outputSingleValuesShuffle = self.ShuffleData(
            inputImages, inputSingleValues, outputSingleValues)
        # Now separate to create batches (i.e., divide the first dimension in 
        # 'number of batches' and 'batch size')
        # The final data attributes are extracted here.
        self.DivideTheDataInBatches(inputImagesShuffle, inputSingleValuesShuffle,
                                    outputSingleValuesShuffle)

        return

        ###########################################################################

    # Functions for defining parameters

    def FindNumberOfImages(self, pathImagesFile):

        imageFiles = os.listdir(pathImagesFile)
        self.numberOfImages = len(imageFiles)

        return

    def FindNumberOfBatches(self):

        # floor, so the last uncomplete batch is thrown out
        self.numberOfBatches = int(np.floor(self.numberOfImages / self.batch_size))

        return

    ###########################################################################
    # Functions for data extraction and reshaping

    def ExtractImagesFromPath(self, pathImagesFile):

        # Preparing the space for the input images
        inputImages = torch.zeros(self.numberOfImages, self.image_channels,
                                  self.x_size, self.y_size)

        # # Define transform
        # transform = transforms.Compose([
        #                transforms.Resize((self.x_size, self.y_size)),
        #                transforms.ToTensor(),
        #             ])

        # Resize without distorsion
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((self.x_size, self.y_size)),
            transforms.ToTensor(),
        ])

        # Extracting the images
        countInImageFolder = 0
        for img in glob.glob(pathImagesFile + "/" + "*.png"):
            imgCurr = Image.open(img)
            imgCurrTransformed = transform(imgCurr)

            # Check resize visually
            # cv2.imshow("", imgCurrTransformed.numpy().transpose(1, 2, 0))
            # cv2.waitKey(0)

            inputImages[countInImageFolder, :, :, :] = imgCurrTransformed

            countInImageFolder += 1

        return inputImages

    # Find min and max value of an annotation in the annotations holder
    @staticmethod
    def FindMinAndMaxInAnnotations(annotationsHolder, annotationName):

        lengthOfAnnotations = len(annotationsHolder.config['annotations'])
        minValue = 100000000
        maxValue = 0
        for i in range(lengthOfAnnotations):

            currentAverageDistance = annotationsHolder.config['annotations'][i][annotationName]
            if currentAverageDistance < minValue:
                minValue = currentAverageDistance
            if currentAverageDistance > maxValue:
                maxValue = currentAverageDistance

        return minValue, maxValue

    def CompactMaxsAndMinsValuesOutputs(self):

        # Save the min and max values of output (to denormalize it after calling the CNN)
        minValues = torch.zeros(self.numberOfOutputSingleValues)
        # minValues[0] = self.minWidthTop
        # minValues[1] = self.minWidthBottom
        minValues[0] = self.minMass
        self.minValuesOutput = minValues

        maxValues = torch.zeros(self.numberOfOutputSingleValues)
        # maxValues[0] = self.maxWidthTop
        # maxValues[1] = self.maxWidthBottom
        maxValues[0] = self.maxMass
        self.maxValuesOutput = maxValues

        return

    def ExtractMinAndMaxValuesOfInputsAndOutputs(self, annotationsHolder):

        # Extracting the min and max of average distance and width, to perform normalization

        # Min and max of average distance, width top and width bottom
        self.minAverageDistance, self.maxAverageDistance = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
                                                                                                    'average distance')
        # self.minWidthTop, self.maxWidthTop = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder, 'width top')
        # self.minWidthBottom, self.maxWidthBottom = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
        #                                                                                     'width bottom')
        self.minMass, self.maxMass = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder, 'mass')

        print("Min mass: {}".format(float(round(self.minMass, 2))))
        print("Max mass: {}".format(float(round(self.maxMass, 2))))

        self.CompactMaxsAndMinsValuesOutputs()

        return

    def ExtractAnnotationInputsAndOutputs(self, annotationsHolder):

        # Preparing the space for the input images, the input values and the output prediction
        inputSingleValues = torch.zeros(self.numberOfImages, self.numberOfInputSingleValues)
        outputSingleValues = torch.zeros(self.numberOfImages, self.numberOfOutputSingleValues)

        # Extracting the annotation inputs and outputs (normalizing them, except for the image ratio)
        for i in range(self.numberOfImages):
            # Extract aspect ratio, average distance (inputs)
            currentAspectRatioWidth = annotationsHolder.config['annotations'][i]['aspect ratio width']
            currentAspectRatioHeight = annotationsHolder.config['annotations'][i]['aspect ratio height']
            currentAverageDistance = annotationsHolder.config['annotations'][i]['average distance']

            # # Extract width width top and bottom (inputs)
            # currentWidthTop = annotationsHolder.config['annotations'][i]['width top']
            # currentWidthBottom = annotationsHolder.config['annotations'][i]['width bottom']
            currentMass = annotationsHolder.config['annotations'][i]['mass']

            # Normalize the data
            currentAverageDistanceNorm = (currentAverageDistance - self.minAverageDistance) / (
                    self.maxAverageDistance - self.minAverageDistance)
            # currentWidthTopNorm = (currentWidthTop - self.minWidthTop) / (self.maxWidthTop - self.minWidthTop)
            # currentWidthBottomNorm = (currentWidthBottom - self.minWidthBottom) / (
            #         self.maxWidthBottom - self.minWidthBottom)
            currentMass = (currentMass - self.minMass) / (self.maxMass - self.minMass)

            # Put the inputs and outputs in their arrays
            inputSingleValues[i, 0] = currentAspectRatioWidth
            inputSingleValues[i, 1] = currentAspectRatioHeight
            inputSingleValues[i, 2] = currentAverageDistanceNorm

            # outputSingleValues[i, 0] = currentWidthTopNorm
            # outputSingleValues[i, 1] = currentWidthBottomNorm
            outputSingleValues[i, 0] = currentMass

        return inputSingleValues, outputSingleValues

    def ShuffleData(self, inputImages, inputSingleValues, outputSingleValues):

        # First we must define the indices of shuffling
        indicesOrder = np.arange(self.numberOfImages)  # All indices in order
        # We shuffle the above indices
        newIndices = indicesOrder
        for i in range(len(indicesOrder)):
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            indicesOrder = np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i] = pickValue

        # Now we shuffle the data based on the above indices
        inputImagesShuffle = inputImages[newIndices, :, :, :]
        inputSingleValuesShuffle = inputSingleValues[newIndices, :]
        outputSingleValuesShuffle = outputSingleValues[newIndices, :]

        return inputImagesShuffle, inputSingleValuesShuffle, outputSingleValuesShuffle

    def DivideTheDataInBatches(self, inputImagesShuffle, inputSingleValuesShuffle,
                               outputSingleValuesShuffle):

        inputImagesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.image_channels,
                                       self.x_size, self.y_size)
        inputSingleValuesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.numberOfInputSingleValues)
        outputSingleValuesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.numberOfOutputSingleValues)

        beginBatch = 0
        endBatch = self.batch_size
        for i in range(self.numberOfBatches):
            inputImagesFinal[i, :, :, :, :] = inputImagesShuffle[beginBatch:endBatch, :, :, :]
            inputSingleValuesFinal[i, :, :] = inputSingleValuesShuffle[beginBatch:endBatch, :]
            outputSingleValuesFinal[i, :, :] = outputSingleValuesShuffle[beginBatch:endBatch, :]

            beginBatch += self.batch_size
            endBatch += self.batch_size

        self.inputImagesBatched = inputImagesFinal
        self.inputSingleValuesBatched = inputSingleValuesFinal
        self.outputSingleValuesBatched = outputSingleValuesFinal

        return
