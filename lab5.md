# Lab 5

Due to ACEHub connection and updating, this lab will focus on implemnting models on your host machine first, and then moving them to ACEHub when it is back up. 
In this lab we will not quite follow the 7-step process because we will be building upon Lab 4 from last week. If your Lab 4 did not work, or you are unsure, ask for help to make sure your base solution works or else you will be fighting problems from two labs at once.

Remember our problem is to identify cat pictures from the CIFAR dataset. We will still have to separate the cats from the other pictures as we did before. 

## Local Setup
The process below may not be the "most idea", but instead is designed to mimic the working conditions and environments you are currently used to. This process only works if you have already installed Git and Docker Desktop, as well as the appropriate extensions in your local VS Code. 

1. Make sure you have your Access token for the ANT GitLab (or make a new one). 
2. Go to https://git.antcenter.net/nyielding/acehub-tensorflow-image
3. Click 'code', and select VS Code (HTTPS). 
4. VS Code will prompt where you would like to clone the repo. Your call. 
5. In the VS Code Explorer (folder structure) for the repo, right click the Dockerfile and select 'Build Image...'
6. In Docker Desktop, your new image should appear. Click Run. 
7. In VS Code, it should show up in the Docker tab. 
8. Right click: 'Open in Window', and select 8888. 
9. Welcome to the good ol environment you know and love. 

Since we aren't working in our 'remote_home' directories, GIT PUSH OFTEN. 

## Lab Files
 - lab5.py: 
 - heatmap.py


## TensorFlow Records
This lab gives example code for using TensorFlow Records (TFRecords) in the transfer learning problem. While we will use the exact same data from the previous lab (the CIFAR-10 dataset) we will assume for some reason we need to read it from disk. The most likely reason for this is that the whole dataset cannot fit in RAM at once. However, we may be able to read our data in small chunks which we convert to TFRecord files. TFRecords allow us to read small chunks of TFRecord files dynamically while training and avoid reading the entire dataset into RAM at once.

Students will take the CIFAR-10 dataset and process it into TFRecords. When writing the TFRecords, only pre-process the labels. Do NOT scale the input data before writing the TFRecords. There should be some split of the records so that there are different dataset objects for training, validation, and testing. It should only be necessary to have about 10-20 TFRecords in total since our dataset is actually pretty small.

Remember you will need to parse the TFRecords into your desired tensor shapes (like for images) and then repeat, shuffle, scale, batch, and prefetch them as necessary for training. Code for accomplishing this is provided. The student must use the TFRecords as the data for training in the transfer learning portion.

## Lab 4 Network
Train your final network from Lab 4 using the TFRecords dataset instead of loading all the files at once to RAM. Verify your performance is about the same as before.

*DELIVERABLE*: Lab4 network retrained using TFRecords and a comparison to your Lab4 performance results

## Pre-Trained Network

We will use a pre-trained network instead of training from scratch as in Lab 4. Load the Keras built in pre-trained CNNs ResNet50 to modify for our cat detector problem. This should involve removing the old output layer and adding a new one to match the output of our current problem.

*DELIVERABLE*: Pre-trained model with clearly defined conv_base section. 

### Real Life Note
I know Keras has CNNs with weights trained for CIFAR-10, so you might be tempted to use the original output of the CNN. However, we are in a learning setting, so we will pretend we cannot do that and must strip off the old output layer and retrain them at least a little. Instead, use the 'imagenet' weights for the pretrained network

### CNN Modification
Take your new CNN and retrain the weights for our problem. This will involve freezing most of the layers close to the input and re-training or maybe even adding new layers to train. I expect that the performance should be better compared to our from-scratch network but this isn't necessary. However, you should at least get similar performance.  

You can decide if you just want to add more layers or retrain existing layers in the model.

Print out the performance of this pre-trained ANN on the CIFAR-10 test set using a TFRecords dataset with. Make sure you use all the samples from the test set even though you are generating them with a TFRecords dataset.  Remember that you should only print the test set when you believe the network is fully trained and will generalize.

*DELIVERABLE*: Final performance of new CNN compared to your Lab4 network. Discuss WHY they performed differently. 

## Visualize Insight for CNNs
This section of the lab is a modified example from the Chollet book. The example can be found [here,](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/660498db01c0ad1368b9570568d5df473b9dc8dd/first_edition/5.4-visualizing-what-convnets-learn.ipynb) and it is highly encouraged students review the example to make code for this lab. This technique uses Class Activation Maps to highlight in the original image the sections the CNN keyed in on to make a classification. Think of this as answering the question "why did you pick that class?". This is a very useful technique for understanding the inner workings of CNNs.

In this section we will explore some visualizations for CNNs. Since everyone has different models that may or may not have trained well, everyone will use a pre-trained model from keras. The template code contains the startup code for the `VGG16` model trained on ImageNet. 

Running the notebook will perform a heat map analysis like the Elephant picture from the Chollet book example. Using the cat/dog images provided make a heat map on what parts of the image the CNN think is cat or like.

The images used are found in the img_in folder. Any images that fit a label from imagenet can be placed in here to try yourself.

The notebook will output the top 3 predicted classes from imagenet and a plot in img_out that overlays the heatmap of the CNN layer activation.

*DELIVERABLE*: In the last cell of your Notebook, discuss the labels guessed for the various images, specifically the differences from Freya and Artemis. 

### Help
Note the example from Chollet is with an older version of TensorFlow that did not have eager execution. In order to run the example (and thus use example code in your project) you must include the line:
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```
to disable eager execution and be able to use the gradient commands the way the example does.
