# Finetuning Mask-RCNN pretrained on MS COCO 

This repo includes functionality to:

#### Test the PyTorch implementation of a pre-trained Mask-RCNN network.

Mask-RCNN is pre-trained on MS COCO and fine-tuned in the final classification layer on:

1. A 200-frame manually labeled custom data set
2. The person subset of the MS COCO 2017 training data set

#### Re-train the final layer of the pre-trained Mask-RCNN network on any custom data set.

The source code is written in python and PyTorch. The state dictionary of the PyTorch based Mask-RCNN model implementation is saved in the maskrcnn.pt file, included in the project folder.

# Installation

Requires python 3.7 and an Nvidia GPU. Tested on Ubuntu 18.4 on an Nvidia GeForce GTX 1660 Ti and an Nvidia GeForce RTX 2080 Super.

Installing dependencies with conda is recommended. A conda environment.yml file is included in the project folder. To install the required dependencies run:


```conda env create -f environment.yml```

If conda is used to install dependencies, the conda environment needs to be activated after its creation as follows:

```conda activate <conda_env_name>```

Install pycocotools (will be used for computing the evaluation metrics following the COCO metric for intersection over union):

```
pip install cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
``` 

Download the TorchVision repo to use some files from references/detection:

```
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0
cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

## Usage

#### To check segmentation performance by feeding sample images to Mask-RCNN run:

```python main.py -image_path <image_path_goes_here>```

Segmentation results will be written in a result.png file in the project root folder.

#### To re-train the final classification layer of Mask-RCNN on a custom data set run:


```python main.py -train_data_path <path to raw color image data used for training> -train_coco_path <path to coco annotations>```
