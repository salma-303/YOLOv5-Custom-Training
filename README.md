# YOLOv5-Custom-Training
This repository contains instructions and code for training the YOLOv5 object detection model on custom data for car detection. YOLOv5 is a popular and efficient deep learning model for real-time object detection, and customizing it to detect cars in your own datasets can be a valuable asset for various applications. Please browse the YOLOv5 [Docs](https://docs.ultralytics.com/yolov5/) for details.

## Prerequisites
Before you begin, make sure you have the following prerequisites installed and set up:

Python: You need Python 3.8 or later. You can download Python from the [official website](https://www.python.org/downloads/).

Git: If you haven't already, install Git by following the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

NVIDIA GPU (optional): Training YOLOv5 is significantly faster on a GPU. If you have access to an NVIDIA GPU, make sure you have installed the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) for GPU support.
Or you can use Free GPU resources like [Google Colab](https://colab.google/).

## Setup 
1. Clone this repository to your local machine:
```ruby
git clone https://github.com/salma-303/YOLOv5-Custom-Training
```
2. Change your working directory to the repository:
```ruby
cd YOLOv5-Custom-Training
```
3. Run YOLOv5 setup cell in `YOLOv5-Car-Detection.ipynb`
```ruby
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt comet_ml  # install

import torch
import utils
display = utils.notebook_init()  # checks
```
4- Label and export your custom datasets directly to YOLOv5 for training with [Roboflow](https://roboflow.com/?ref=ultralytics).
In this repo, I trained the model to detect cars, you can find the labeled data set [here](https://universe.roboflow.com/skyxperts/yolov5-car-detection).
To use it in training, run this:
```ruby
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR API KEY")
project = rf.workspace("skyxperts").project("yolov5-car-detection")
dataset = project.version(2).download("yolov5")
```
The dataset includes 3080 images.
Cars are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create three versions of each source image:
* Random rotation of between -15 and +15 degrees
* Random brightness adjustment of between -25 and +25 percent

The following transformations were applied to the bounding boxes of each image:
* 50% probability of vertical flip
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 2.5 pixels

## Training
To train YOLOv5 on your custom dataset for car detection, follow these steps:

1. Modify the `data.yaml` file in the yolov5-custom-car-detection/data directory with the following information:
```ruby
test: ../test/images                          # Path to your testing data
train: YOLOv5-Car-detection--2/train/images   # Path to your training data
val: YOLOv5-Car-detection--2/valid/images     # Path to your validation data
nc: 1                         # Number of classes (1 for car detection)
names: ['car']                # Class names
```
2. Run the training script:
```ruby
python train.py --img 640 --epochs 120 --data data.yaml --weights yolov5s.pt --cache ram
```
You can customize the training parameters as you need.
If you have pre-trained weights, specify them with the --weights flag.
Add `--cache ram` or `--cache disk` to speed up training (requires significant RAM/disk resources).

3. After training, you can evaluate your model on a test dataset:

```ruby
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt
```
5. To perform inference on new images, use the following command:
```ruby
python detect.py --weights runs/train/exp/weights/best.pt --img-size 640 --source path/to/your/image.jpg
```
I provided my pre-trained model on car detection using YOLOv5s, you can find it in `train` folder.

## Acknowledgments
This repository is based on the official [YOLOv5 repository](https://github.com/ultralytics/yolov5). Please take a look at their documentation for additional information and advanced usage.

## Conclusion
You now have the basic setup and instructions for training YOLOv5 on custom data for car detection. Feel free to adapt and extend this repository to suit your specific needs and datasets. Good luck with your object detection project!
