""" This script generates the final predictions.
In the following the steps:
1. From every video, sample a frame every tot (based on the parameter).
2. For every frame run instance segmentation model and retrieve bounding boxes, masks, scores for the selected classes. Discard frames without detections.
3. For each detected object select the k nearest objects
4. Run Encoder on the last k nearest objects and average the predictions.
"""

import argparse
import torch
import numpy as np
import cv2
import torchvision
import os
import time
import pandas as pd
from Encoder.Models import ConfigurationHolder as CH
from Encoder.Dataset.select_last_k_frames import SelectLastKFrames
from Encoder.Models import CNN_encoder
from utils.csv_utils import CsvResults
from utils.utils import get_outputs, get_filenames, filter_results
from torchvision.transforms import transforms as transforms
from utils.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from Encoder.Models.DataExtractor import SquarePad
from PIL import Image
import torchvision.transforms.functional as F

transform_enc = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224), interpolation=F.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def compute_average(path_to_gt, col_name):
    gt = pd.read_csv(path_to_gt, sep=',')
    cc_avg = np.average(gt[col_name].unique())
    return int(cc_avg)


def generate_data(path_to_video_dir, path_to_dpt_dir, path_to_ann):
    # TODO: set offset
    offset = 1  # Sample one every offset frames
    # TODO:set mins and maxs
    minAverageDistance, maxAverageDistance, minMass, maxMass = [276.0, 1534.06, 2.0, 134.0]
    minValuesOutput = torch.tensor(minMass)
    maxValuesOutput = torch.tensor(maxMass)

    # initialize the model
    segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                                            num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    segmentation_model.to(device).eval()

    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # retrieve file names
    path_to_rgb_video = get_filenames(path_to_video_dir, ".mp4")
    path_to_rgb_video.sort()
    project_dir = os.path.dirname(os.path.dirname(__file__))
    pathConfigurationFile = os.path.join(project_dir, 'Encoder/Configuration/ConfigurationFile.json')
    configHolder = CH.ConfigurationHolder()
    configHolder.LoadConfigFromJSONFile(pathConfigurationFile)

    # Initialize CNN and print it
    encoder = CNN_encoder.CNN_encoder(image_size=configHolder.config['x_size'],
                                      dim_filters=configHolder.config['dim_filters'],
                                      kernel=configHolder.config['kernels_size'],
                                      stride=configHolder.config['stride'],
                                      number_of_neurons_middle_FC=configHolder.config['number_of_neurons_middle_FC'],
                                      number_of_neurons_final_FC=configHolder.config['number_of_neurons_final_FC'],
                                      number_of_cameras=configHolder.config['number_of_cameras'],
                                      minValuesOutput=minValuesOutput,
                                      maxValuesOutput=maxValuesOutput)
    encoder.load_state_dict(
        torch.load(os.path.join(project_dir, "Encoder/CNN_27.torch")))
    encoder.eval()
    algo = SelectLastKFrames()
    csv_res = CsvResults()
    # loop through the videos
    for ind in range(0, len(path_to_rgb_video)):
        start_time = time.time()
        print(os.path.basename(path_to_rgb_video[ind]))
        video_cap = cv2.VideoCapture(path_to_rgb_video[ind])
        counter = 0
        # retrieve correct depth frames
        depth_frames = get_filenames(
            os.path.join(path_to_dpt_dir, os.path.basename(path_to_rgb_video[ind].rsplit('.', 1)[0])), ".png")
        depth_frames.sort()

        rgb_patches_list = []
        dpt_patches_list = []
        prediction_list = []
        mask_patches_list = []
        while video_cap.isOpened():
            ret, bgr_frame = video_cap.read()

            # if frame is correctly retrieved
            if ret:
                # convert to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                # orig_frame = bgr_frame.copy()
                # transform the image
                rgb_frame = transform(rgb_frame)
                # add a batch dimension
                rgb_frame = rgb_frame.unsqueeze(0).to(device)

                # perform object prediction
                masks, boxes, cls, scores = get_outputs(rgb_frame, segmentation_model, 0.4)
                masks, boxes, cls, scores = filter_results(masks, boxes, cls, scores)
                # result = draw_segmentation_map(orig_frame, masks, boxes, cls, scores)
                # visualize the image
                # cv2.imshow('Segmented image', result)

                # load depth image
                dpt_im = cv2.imread(depth_frames[counter], -1)
                im = cv2.cvtColor(bgr_frame.copy(), cv2.COLOR_BGR2RGB)
                # cv2.imshow("depth", dpt_im)
                # iterate through detections in the current frame
                for i in range(0, len(cls)):
                    [xmin, ymin] = boxes[i][0]
                    [xmax, ymax] = boxes[i][1]
                    [xmin, ymin, xmax, ymax] = [int(xmin), int(ymin), int(xmax), int(ymax)]
                    # print("bbox: [{}, {}, {}, {}]".format(xmin, ymin, xmax, ymax))
                    # print("class: {}".format(cls[i]))

                    # compute aspect ratio height
                    ar_h = round((ymax - ymin) / im.shape[0], 2)
                    # compute aspect ratio width
                    ar_w = round((xmax - xmin) / im.shape[1], 2)
                    # print("aspect ratio width: {}".format(ar_w))
                    # print("aspect ratio height: {}".format(ar_h))

                    # compute average distance considering only segmentation pixels in depth map
                    mask = masks[i]
                    mask = mask[ymin:ymax, xmin:xmax]
                    temp_dpt = dpt_im.copy()
                    patch_dpt = temp_dpt[ymin:ymax, xmin:xmax]
                    dpt_patches_list.append(patch_dpt)
                    mask_patches_list.append(mask)
                    # cv2.imshow("Patch depth", patch_dpt)
                    non_zero_count = np.count_nonzero(patch_dpt[mask])
                    avg_d = round(np.sum(patch_dpt[mask]) / non_zero_count, 2)
                    score_temp = round(scores[i], 2)
                    prediction_list.append((cls[i], score_temp, ar_w, ar_h, avg_d))

                    # print("average distance: {}".format(avg_d))

                    # provide feedback
                    print("Class: {}  \t confidence: {:.2}  \t average distance: {:.2f}mm".format(coco_names[cls[i]],
                                                                                                  scores[i], avg_d))

                    # crop RGB
                    temp_rgb = im.copy()
                    patch_rgb = temp_rgb[ymin:ymax, xmin:xmax]
                    rgb_patches_list.append(patch_rgb)

                    # cv2.imshow("Patch rgb", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(0)
                # cv2.waitKey(1)

                # skip to the next frame based on the offset value
                counter += offset
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            else:
                break
        algo.update_frames(rgb_patches_list, dpt_patches_list, prediction_list, mask_patches_list)
        algo.select_frames()

        csv_res.fill_entry('Configuration ID', int(os.path.splitext(os.path.basename(path_to_rgb_video[ind]))[0]))
        pred = None
        if len(algo.selected_rgb_patches) == 0 or \
                len(algo.selected_dpt_patches) == 0 or \
                len(algo.selected_predictions) == 0 or \
                len(algo.selected_mask_patches) == 0:
            pred = compute_average(path_to_ann, 'container mass')
        else:
            # Retrieve images
            images = algo.selected_rgb_patches
            k = len(images)
            inputImages = torch.zeros(k, 3,
                                      224, 224)
            inputSingleValues = torch.zeros(k, 3)
            for j in range(0, k):
                imagesCurr = Image.fromarray(images[j])
                imgCurrTransformed = transform_enc(imagesCurr)
                # cv2.imshow("", imgCurrTransformed.numpy().transpose(1, 2, 0)[:,:,::-1])
                # cv2.waitKey(0)
                inputImages[j, :, :, :] = imgCurrTransformed
                # Retrieve input values
                [_, _, ar_w, ar_h, avg_d] = algo.selected_predictions[j]

                inputSingleValues[j, 0] = ar_w
                inputSingleValues[j, 1] = ar_h
                inputSingleValues[j, 2] = (avg_d - minAverageDistance) / (maxAverageDistance - minAverageDistance)
                # perform final prediction
                predictedValues = encoder(inputImages, inputSingleValues)
                pred = encoder.AveragePredictions(encoder.CalculateOutputValueDenormalized(predictedValues, k))
        csv_res.fill_entry('Container mass', int(pred))
        csv_res.fill_other_entries(['Container capacity',  # 'Container mass',\
                                    'Filling mass', 'None', 'Pasta', 'Rice', 'Water', 'Filling type', 'Empty', \
                                    'Half-full', 'Full', 'Filling level', 'Width at the top', 'Width at the bottom', \
                                    'Height', 'Object safety', 'Distance', 'Angle difference'], -1)
        elapsed_time = time.time() - start_time
        csv_res.fill_entry('Execution time', round(elapsed_time,2))
        video_cap.release()

    csv_res.save_csv("predictions.json")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='generate_training_set',
                                     usage='%(prog)s --path_to_video_dir <PATH_TO_VIDEO_DIR> --path_to_dpt_dir <PATH_TO_DPT_DIR> --path_to_ann <PATH_TO_ANN>')
    parser.add_argument('--path_to_video_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/rgb")
    parser.add_argument('--path_to_dpt_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/depth")
    parser.add_argument('--path_to_ann', type=str,
                        default="/home/sealab-ws/PycharmProjects/Corsmal_challenge/CORSMALChallengeEvalToolkit-master/annotations/ccm_train_annotation.csv")
    args = parser.parse_args()

    # Assertions
    assert args.path_to_video_dir is not None, "Please, provide path to video directory"
    assert os.path.exists(args.path_to_video_dir), "path_to_video_dir does not exist"
    assert args.path_to_dpt_dir is not None, "Please, provide path to depth frames directory"
    assert os.path.exists(args.path_to_dpt_dir), "path_to_dpt_dir does not exist"
    assert args.path_to_ann is not None, "Please, provide path to annotation file"
    assert os.path.exists(args.path_to_ann), "path_to_ann does not exist"

    # Generate data
    generate_data(args.path_to_video_dir, args.path_to_dpt_dir, args.path_to_ann)

    print('Finished!')
