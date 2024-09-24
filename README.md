# Developing a Handwritten Digits Deep Learning Classifier with PyTorch

In this project, I developed and trained a multi-layer perceptron or feedforward neural network for classifying the MNIST handwritten digit dataset using PyTorch. I achieved an accuracy of 97% after training only for 5 epochs. 

## Dataset

The MNIST handwriiten digit dataset MNIST is included in the torchvision module. You can create your dataset using the MNIST object from torchvision.datasets

## Methods

The methods employed in this project involve:
* Downloading, preprocessing, and exploring the images
* Building and training the feedforward neural network classifier
* Evaluating and saving the feedforward neural network classifier

## Network Architecture

The network architecture consists of the input layer (containing 784 nodes representing the size of an input image), 2 hidden layers (the first and second containing 100 and 50 nodes respectively) and the output layer (containing 10 nodes representing the number of classes). ReLu activation and dropout were used during training.

## Requirements

* opencv-python-headless==4.5.3.56
* matplotlib==3.4.3
* numpy==1.21.2
* pillow==7.0.0
* bokeh==2.1.1
* torch==1.11.0
* torchvision==0.12.0
* tqdm==4.63.0
* ipywidgets==7.7.0
* livelossplot==0.5.4
* pytest==7.1.1
* pandas==1.3.5
* seaborn==0.11.2
* jupyter==1.0.0
* ipykernel==4.10.0

## GPU

The notebook is set up to use GPU if it is available.

## Instructions 

* Before running the jupyter notebook, users must have the requirements stored in "requirements.txt" file in the same directory as the jupyter notebook.

## Acknowledgements

I am grateful to AWS for awarding me the AWS AI & Machine Learning Scholarhip to pursue the nanodegree in "Machine Learning Fundamentals" through Udacity. This Jupyter notebook represents the second out of four projects I undertook as part of the requirements to complete the nanodegree.