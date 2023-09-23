# YOLOv5-Custom-Training
This repository contains instructions and code for training the YOLOv5 object detection model on custom data for car detection. YOLOv5 is a popular and efficient deep learning model for real-time object detection, and customizing it to detect cars in your own datasets can be a valuable asset for various applications. Please browse the YOLOv5 [Docs](https://docs.ultralytics.com/yolov5/) for details.

## Prerequisites
Before you begin, make sure you have the following prerequisites installed and set up:

Python: You need Python 3.8 or later. You can download Python from the [official website](https://www.python.org/downloads/).

Git: If you haven't already, install Git by following the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

NVIDIA GPU (optional): Training YOLOv5 is significantly faster on a GPU. If you have access to an NVIDIA GPU, make sure you have installed the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) for GPU support.
Or you can use Free GPU resources like [Google Colab](https://colab.google/).

## Setup 
1- Clone this repository to your local machine:
```ruby
git clone https://github.com/salma-303/YOLOv5-Custom-Training
```
2- Change your working directory to the repository:
```ruby
cd YOLOv5-Custom-Training
```
3- Run YOLOv5 setup cell in `YOLOv5-Car-Detection.ipynb`
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
