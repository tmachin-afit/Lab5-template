# Lab 5
In this lab we will not quite follow the 7-step process because we will be building upon Lab 4 from last week. If your Lab 4 did not work, or you are unsure, ask for help to make sure your base solution works or else you will be fighting problems from two labs at once.

Remember our problem is to identify cat pictures from the CIFAR dataset. We will still have to separate the cats from the other pictures as we did before. 

## TensorFlow Records
This lab will test your ability to use TensorFlow Records (TFRecords). While we will use the exact same data from the previous lab (the CIFAR-10 dataset) we will assume for some reason we need to read it from disk. The most likely reason for this is that the whole dataset cannot fit in RAM at once. However, we may be able to read our data in small chunks which we convert to TFRecord files. TFRecords allow us to read small chunks of TFRecord files dynamically while training and avoid reading the entire dataset into RAM at once. 

Students will take the CIFAR-10 dataset and process it into TFRecords. When writing the TFRecords, only pre-process the labels. Do *NOT* scale the input data before writing the TFRecords. There should be some split of the records so that there are different dataset objects for training, validation, and testing. It should only be necessary to have about 10-20 TFRecords in total since our dataset is actually pretty small.

Remember you will need to parse the TFRecords into your desired tensor shapes (like for images) and then repeat, shuffle, scale, batch, and prefetch them as necessary for training. 

## Lab 4 Network
Train your final network from Lab 4 using the TFRecords dataset instead of loading all the files at once to RAM. Verify your performance is about the same as before.

## Pre-Trained Network

Now that we have tried training a network from scratch we will use a pre-trained network. Pick one of the Keras built in pre-trained CNNs to modify for our cat detector problem. This should involve removing the old output layer and adding a new one to match the output of our current problem.

### Real Life Note
I know Keras has CNNs with weights trained for CIFAR-10, so you might be tempted to use the original output of the CNN. However, we are in a learning setting, so we will pretend we cannot do that and must strip off the old output layer and retrain them at least a little.

### CNN Modification
Take your CNN of choice and retrain the weights for our problem. This will involve freezing most of the layers close to the input and re-training or maybe even adding new layers to train. I expect that the performance should be better compared to our from-scratch network but this isn't necessary. However, you should at least get similar performance.  

You can decide if you just want to add more layers or retrain existing layers in the model.

Print out the performance of this pre-trained ANN on the CIFAR-10 test set using a TFRecords dataset with. Make sure you use all the samples from the test set even though you are generating them with a TFRecords dataset.  Remember that you should only print the test set when you believe the network is fully trained and will generalize.

## Visualize Insight for CNNs
This section of the lab is a modified example from the Chollet book. The example can be found [here,](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/660498db01c0ad1368b9570568d5df473b9dc8dd/first_edition/5.4-visualizing-what-convnets-learn.ipynb) and it is highly encouraged students review the example to make code for this lab. This technique uses Class Activation Maps to highlight in the original image the sections the CNN keyed in on to make a classification. Think of this as answering the question "why did you pick that class?". This is a very useful technique for understanding the inner workings of CNNs.

In this section we will explore some visualizations for CNNs. Since everyone has different models that may or may not have trained well, everyone will use a pre-trained model from keras. The template code contains the startup code for the `VGG16` model trained on ImageNet. 

You will perform a heat map analysis like the Elephant picture from the Chollet book example. Using the cat images provided make a heat map on what parts of the image the CNN think is cat like.

The two cat images to use are found in the template for this lab in the `src` folder. The images are called `artemis.jpg` and `freya.jpg`. 

Print out the top three predicted classes are for each picture. Compare this to where the heatmap is highlighting the image. Write observation in a python comment with a note for which image you are writing about. Remember to apply the same preprocessing as your selected model did using the keras functions supplied for that model.  

Make a single figure with subplots of the three images: the original image, the heatmap, and the heatmap overlaid on the original image. These images may not be the same number of pixels which is alright, however, make sure they are all readable in the figure.  

Save the heatmap overlay images separately by appending `heatmap_` to the front of the filename and include it in your git repository. Usually you would not include output data like this in the repository, but we will make an exception for class, so I can grade the heatmaps.

### Help
Note the example from Chollet is with an older version of TensorFlow that did not have eager execution. In order to run the example (and thus use example code in your project) you must include the line:
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```
to disable eager execution and be able to use the gradient commands the way the example does.
