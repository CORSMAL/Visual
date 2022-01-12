# This is the File for training the CNN encoder.

###############################################################################
# Importing necessary libraries

import os
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms

from Encoder.Models import ConfigurationHolder as CH
from Encoder.Models import CNN_encoder
from Encoder.Models import DataExtractor as DE

###############################################################################
from Encoder.Models.DataExtractor import SquarePad
from utils.annotation_parser import JsonParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# Retrieving path to current folder and to configuration file.
# Configuration file contains all the information regarding parameters.

# Current folder
pathCurrentFolder = "/home/sealab-ws/PycharmProjects/Apicella_PyCharm/Visual/Encoder"
# Path to configuration file (containing all paramters of nets)
pathConfigurationFile = pathCurrentFolder + '/Configuration/ConfigurationFile.json'
configHolder = CH.ConfigurationHolder()
configHolder.LoadConfigFromJSONFile(pathConfigurationFile)
# Path to locations file (containing the paths to the inputs and if to use GPU/CPU)
pathLocationsFile = pathCurrentFolder + '/Configuration/LocationsFile.json'
locationsHolder = CH.ConfigurationHolder()
locationsHolder.LoadConfigFromJSONFile(pathLocationsFile)

# Path to output Folder
outputFolder = locationsHolder.config['output_folder']

# Take out the path to the annotations datasets
pathAnnotationsFileTraining = locationsHolder.config['annotations_folder_training']
pathAnnotationsFileValidation = locationsHolder.config['annotations_folder_validation']
# Create a configuration holder for the annotations
annotationsHolderTraining = CH.ConfigurationHolder()
annotationsHolderTraining.LoadConfigFromJSONFile(pathAnnotationsFileTraining)
annotationsHolderValidation = CH.ConfigurationHolder()
annotationsHolderValidation.LoadConfigFromJSONFile(pathAnnotationsFileValidation)

# Take out the path to the images
pathImagesFileTraining = locationsHolder.config['images_folder_training']
pathImagesFileValidation = locationsHolder.config['images_folder_validation']

###############################################################################
# Creating data holders for training and validation data
# Training
dataExtractorTraining = DE.DataExtractor(annotationsHolderTraining, pathImagesFileTraining, configHolder)
# Take the mins and maxs of training and pass them to normalize the validation
trainingMinsAndMaxs = dataExtractorTraining.minAverageDistance, dataExtractorTraining.maxAverageDistance, \
                      dataExtractorTraining.minMass, dataExtractorTraining.maxMass
                      # dataExtractorTraining.minWidthTop, dataExtractorTraining.maxWidthTop, \
                      # dataExtractorTraining.minWidthBottom, dataExtractorTraining.maxWidthBottom, \
                      # dataExtractorTraining.minHeight, dataExtractorTraining.maxHeight

# Validation/Testing
dataExtractorValidation = DE.DataExtractor(annotationsHolderValidation, pathImagesFileValidation, configHolder,
                                           trainingMinsAndMaxs)

###############################################################################

# Initialize CNN and print it
CNN = CNN_encoder.CNN_encoder(image_size=configHolder.config['x_size'],
                              dim_filters=configHolder.config['dim_filters'],
                              kernel=configHolder.config['kernels_size'],
                              stride=configHolder.config['stride'],
                              number_of_neurons_middle_FC=configHolder.config['number_of_neurons_middle_FC'],
                              number_of_neurons_final_FC=configHolder.config['number_of_neurons_final_FC'],
                              number_of_cameras=configHolder.config['number_of_cameras'],
                              minValuesOutput=dataExtractorTraining.minValuesOutput,
                              maxValuesOutput=dataExtractorTraining.maxValuesOutput)
CNN.print()

CNN.load_state_dict(torch.load("/home/sealab-ws/PycharmProjects/Apicella_PyCharm/Visual/Encoder/OUTPUTS_0_adam/CNN_74.torch"))
CNN.eval()

ann = JsonParser()
ann.load_json(pathAnnotationsFileValidation)
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224), interpolation=F.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
for i in range(0, len(ann.image_name)):
    file_name = ann.image_name[i]
    imgCurr = Image.open(os.path.join(pathImagesFileValidation, file_name))

    imgCurrTransformed = transform(imgCurr)
    # cv2.imshow("", imgCurrTransformed.numpy().transpose(1, 2, 0)[:,:,::-1])
    # cv2.waitKey(0)

    inputImages = torch.zeros(1, 3,
                              224, 224)
    inputImages[0,:,:,:] = imgCurrTransformed

    inputSingleValues = torch.zeros(1, 3)
    inputSingleValues[0, 0] = ann.ar_w[i]
    inputSingleValues[0, 1] = ann.ar_h[i]
    inputSingleValues[0, 2] = (ann.avg_d[i] - dataExtractorValidation.minAverageDistance) / (dataExtractorValidation.maxAverageDistance - dataExtractorValidation.minAverageDistance)
    predictedValues = CNN(inputImages, inputSingleValues)
    annSingleValues = torch.zeros(1, 3)
    annSingleValues[0,0] = ann.mass[i]
    # outputSingleValues[0,0] = (ann.wt[i] - dataExtractorValidation.minWidthTop) / (dataExtractorValidation.maxWidthTop - dataExtractorValidation.minWidthTop)
    # outputSingleValues[0,1] = (ann.wb[i] - dataExtractorValidation.minWidthBottom) / (
    #         dataExtractorValidation.maxWidthBottom - dataExtractorValidation.minWidthBottom)
    # outputSingleValues[0,2] = (ann.height[i] - dataExtractorValidation.minHeight) / (
    #         dataExtractorValidation.maxHeight - dataExtractorValidation.minHeight)
    print("true: {}".format(ann.mass[i]))
    print("prediction: {}".format(int(max(0,CNN.CalculateOutputValueDenormalized(predictedValues, 1).cpu().detach().numpy()[0][0]))))
