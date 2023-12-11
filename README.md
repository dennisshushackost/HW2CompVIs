# Homework 2: Fully Convolutional Networks

## Introduction

In this exercise, you should design and train a convolutional neural network to classify the pixels of an image into the different classes of the cityscape data set that is used for training autonomous driving.

## Data Preparation
The whole data set is quite large, so a smaller portion of the data set has already been prepared and divided into smaller image patches of 256x256 patches. (There are 2 versions, the normal train set which uses 3 cities, and a smaller one called train_small that only uses one city).

## ConvNet Approaches
The task is to implement two different fully convolutional neural networks and train and validate them on the data:

For the first network, a relatively simple FCN should be used that only uses convolutional layers of stride 1, but no downsampling of the data. This is for testing the overall approach and that the training procedure has not errors.
The second network should use a some form of downsampling and upsampling. Try to tune the network to get the best results and try different network architectures.
Conv nets approaches benefit a lot from gpus. If you don't have a gpu available, use more simple networks to get first results and also keep the networks for the second task smaller.

Use the train and validation data sets to tune your hyper parameters. This includes changing the number of layers or the sizes of the convolutional filters in the network.

Finally run the networks on the test data set to compare the performance

Tips
While you can try to train the network in a python notebook, you could also consider to run it as a standalone program, as training takes some time

In both cases, you might want to use tensorboard for visualisation (will be quickly explained further in lesson 10) or wandb (https://wandb.ai/home)

Start with a small network and see if it converges and then enlarge it by adding more layers etc.

Test the network first with the example data sets from exercise 9.

## Datasets
The data sets are available from https://drive.switch.ch/index.php/s/PfgmHsurnnsU36t

Please only use the dataset for the course. If you are interested in the complete data set, please register at https://www.cityscapes-dataset.com/