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

from select_last_k_frames import SelectLastKFrames
from utils.original_annotations_parser import JsonParser
from utils.csv_utils import CsvResults
from utils.utils import get_outputs, get_filenames, filter_results
from torchvision.transforms import transforms as transforms
from utils.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names



def generate_data(path_to_video_dir, path_to_dpt_dir, path_to_dest):
    # TODO: set offset
    offset = 1  # Sample one every offset frames

    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()

    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # retrieve file names
    path_to_rgb_video = get_filenames(path_to_video_dir, ".mp4")
    path_to_rgb_video.sort()

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
                masks, boxes, cls, scores = get_outputs(rgb_frame, model, 0.4)
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
        # perform final prediction
        pred = algo.final_prediction()
        csv_res.fill_entry('Container mass', pred)
        csv_res.fill_other_entries(['Container capacity',  # 'Container mass',\
                                    'Filling mass', 'None', 'Pasta', 'Rice', 'Water', 'Filling type', 'Empty', \
                                    'Half-full', 'Full', 'Filling level', 'Width at the top', 'Width at the bottom', \
                                    'Height', 'Object safety', 'Distance', 'Angle difference'], -1)
        elapsed_time = time.time() - start_time
        csv_res.fill_entry('Execution time', elapsed_time)
        video_cap.release()

    csv_res.save_csv(os.path.join(path_to_dest, "annotations.json"))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='generate_training_set',
                                     usage='%(prog)s --path_to_video_dir <PATH_TO_VIDEO_DIR> --path_to_dpt_dir <PATH_TO_DPT_DIR>')
    parser.add_argument('--path_to_video_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/rgb")
    parser.add_argument('--path_to_dpt_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/depth")
    parser.add_argument('--path_to_annotations', type=str,
                        default="/home/sealab-ws/PycharmProjects/Corsmal_challenge/CORSMALChallengeEvalToolkit-master/annotations/ccm_train_annotation.json")
    parser.add_argument('--path_to_dest_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito")
    args = parser.parse_args()

    # Assertions
    assert args.path_to_video_dir is not None, "Please, provide path to video directory"
    assert os.path.exists(args.path_to_video_dir), "path_to_video_dir does not exist"
    assert args.path_to_dpt_dir is not None, "Please, provide path to depth frames directory"
    assert os.path.exists(args.path_to_dpt_dir), "path_to_dpt_dir does not exist"
    assert args.path_to_dest_dir is not None, "Please, provide path to destination directory"
    assert os.path.exists(args.path_to_dest_dir), "path_to_dest_dir does not exist"

    # Generate data
    generate_data(args.path_to_video_dir, args.path_to_dpt_dir, args.path_to_dest_dir)

    print('Finished!')
