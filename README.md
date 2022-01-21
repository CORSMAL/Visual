# Corsmal Challenge - Team Visual 

This repository contains the methodology proposed by Visual team in CORSMAL challenge.

We propose a method to provide an estimation of the container mass (Task 4) exploiting RGB-D data coming from the view 3 (from the robot perspective), using a two-stage pipeline. The first stage employs a detection and segmentation network to locate the container. The second part uses a simple and lightweight encoder to provide the actual mass estimation. 
 
A brief description of the method:
1. For each video, every frame is sampled and the object detection and segmentation is performed using Mask R-CNN model pretrained on COCO.
2. Leveraging the average distance, computed considering the depth map only in the pixels positions belonging to the segmentation mask, we select the 5 nearest objects (least average distance with respect to the camera of the chosen view). 
3. The final prediction of the container mass is the average of the 5 predictions (one per each nearest detected object) performed by a lightweight CNN encoder model.

## Table of contents
* [Installation](#installation)
* [Instructions](#instructions)
* [Demo](#demo)
* [Data format](#data-format)
* [Contacts](#contacts)
* [License](#license)

## Installation

### Setup specifics
* *OS:* Ubuntu 20.04.3 LTS
* *Kernel version:* 5.11.0-46-generic
* *CPU:* Intel® Core™ i9-9900 CPU @ 3.10GHz
* *Cores:* 16 
* *RAM:* 32 GB
* *GPU:* NVIDIA GeForce RTX 2080 Ti

### Requirements
The name of the main libraries and their versions are reported in the following list:
* pytorch=1.10.1
* torchvision=0.11.2
* scipy=1.7.3
* matplotlib=3.5.1 
* torchsummary=1.5.1
* pandas=1.3.5
* opencv=4.5.2

The file *requirements.txt* reports all libraries and their versions. To install them the following code snippet can be used:

    # Create conda environment
    conda create --name CCM python=3.8 # or conda create -n CCM python=3.8
    conda activate CCM
    
    # Install libraries
    pip install torch torchvision scipy matplotlib torchsummary pandas
    conda install -c conda-forge opencv



## Instructions
0. Clone the repository
1. Install the requirements
2. Run *demo/generate_video_inference.py* passing as arguments the path to the directory of RGB (.mp4) videos and depth files (.png).  

## Demo
*demo/generate_video_inference.py* runs the demo of the proposed method and creates the submission .csv file.

### Running arguments
The running arguments of the python demo are:
* `path_to_video_dir`: path to the directory containing RGB .mp4 videos
* `path_to_dpt_dir`: path to the directory containing the depth .png images
These arguments are loaded as strings, hence the inverted commas must be used e.g. "home/dataset/rgb_images".

### Running examples
    # Run demo
    python demo/generate_video_inference.py --path_to_video_dir <PATH_TO_VIDEO_DIR> --path_to_dpt_dir <PATH_TO_DPT_DIR>  


## Data format
### Input
The proposed method uses both RGB and depth images, in particular: the detection/segmentation model uses RGB, the 5 candidate selection uses RGB, masks, depth, and the encoder model uses RGB images.  

### Output
The output of final stage of the encoder is a float value in range [0, 1]. In the demo is shown how to provide the output in the appropriate range.

## Contacts
If you have any further enquiries, question, or comments, please contact <email>XXXX</email>. 
If you would like to file a bug report or a feature request, use the Github issue tracker. 


## License
This work is licensed under the MIT License.  To view a copy of this license, see
[LICENSE](LICENSE).

