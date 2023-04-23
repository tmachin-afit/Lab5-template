import datetime
import itertools
import os
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm
from tqdm.keras import TqdmCallback


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# this function was added since the _int64_feature takes a single int and not a list of ints
def _int64_list_feature(value: typing.List[int]):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert(x_data: np.ndarray,
            y_data: np.ndarray,
            out_path: Path,
            records_per_file: int = 4500,
            ) -> typing.List[Path]:
    """
    Function for reading images from disk and writing them along with the class-labels to a TFRecord file.

    :param x_data: the input, feature data to write to disk
    :param y_data: the output, label, truth data to write to disk
    :param out_path: File-path for the TFRecords output file.
    :param records_per_file: the number of records to use for each file
    :return: the list of tfrecord files created
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Open a TFRecordWriter for the output-file.
    record_files = []
    # TODO: make the tf records here and return the files that were created
    return record_files


def convert_datasets(base_dir: Path):
    # label_names = {
    #     0: 'airplane',
    #     1: 'automobile',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer',
    #     5: 'dog',
    #     6: 'frog',
    #     7: 'horse',
    #     8: 'ship',
    #     9: 'truck',
    # }
    cat_indices = [3]
    (x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

    base_train_dir = base_dir / 'train'
    base_test_dir = base_dir / 'test'

    y_train_cats = np.array(np.isin(y_train_raw, cat_indices), dtype=int)
    y_test_cats = np.array(np.isin(y_test_raw, cat_indices), dtype=int)

    train_files = convert(x_data=x_train, y_data=y_train_cats, out_path=base_train_dir)
    test_files = convert(x_data=x_test, y_data=y_test_cats, out_path=base_test_dir)

    return train_files, test_files


def get_dataset(filenames: typing.List[Path],
                img_shape: tuple) -> tf.data.Dataset:
    """
    This function takes the filenames of tfrecords to process into a dataset object
    The _parse_function takes a serialized sample pulled from the tfrecord file and
    parses it into a sample with x (input) and y (output) data, thus a full sample for training

    This function will not do any scaling, batching, shuffling, or repeating of the dataset

    :param filenames: the file names of each tf record to process
    :param img_shape: the size of the images a width, height, channels
    :return: the dataset object made from the tfrecord files and parsed to return samples
    """

    def _parse_function(serialized):
        # TODO parse the serialized data into x and y samples here
        x_sample = None
        y_sample = None
        return x_sample, y_sample

    # the tf functions takes string names not path objects, so we have to convert that here
    filenames_str = [str(filename) for filename in filenames]
    # make a dataset from slices of our file names
    files_dataset = tf.data.Dataset.from_tensor_slices(filenames_str)

    # make an interleaved reader for the TFRecordDataset files
    # this will give us a stream of the serialized data interleaving from each file
    dataset = files_dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x),
                                       cycle_length=len(filenames),  # how many files to cycle through at once
                                       block_length=1,  # how many samples from each file to get
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       deterministic=False)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(map_func=_parse_function,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def visualize_model(model: Model,
                    x_visualize: np.ndarray,
                    y_visualize: np.ndarray):
    """
    Visualize our predictions using classification report and confusion matrix

    :param model: the model used to make predictions for visualization
    :param x_visualize: the input features given used to generate prediction
    :param y_visualize: the true output to compare against the predictions
    """
    y_pred = model.predict(x_visualize)
    y_pred = np.array(y_pred > 0.5, dtype=int)
    y_true = y_visualize
    class_names = ['not cat', 'cat']

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


def visualize_model_dataset(model: Model,
                            dataset: tf.data.Dataset,
                            eval_samples=6000):
    """
    Uses a model and a dataset to visualize the current predictions

    :param model: the model used to make predictions
    :param dataset: the dataset to use for generating predictions and comparing against truth
    :param eval_samples: the number of samples used to evaluate from the dataset
    """
    x_visualize = []
    y_visualize = []

    total_samples = 0
    img_batch: np.ndarray
    label_batch: np.ndarray
    for img_batch, label_batch in dataset.as_numpy_iterator():
        total_samples += img_batch.shape[0]
        x_visualize.append(img_batch)
        y_visualize.append(label_batch[:, None])
        if total_samples > eval_samples:
            break

    x_visualize = np.vstack(x_visualize)
    y_visualize = np.vstack(y_visualize)

    visualize_model(model, x_visualize, y_visualize)


def main():
    base_dir = Path('/opt', 'data', 'cifar10-tfrecords', 'yourname')

    force_make_tfrecords = True
    visualize_dataset = False

    data_img_shape = (32, 32, 3)

    train_model = True
    train_pre_trained_model = True

    show_metrics_for_valid_dataset = True

    # get tfrecord files
    if force_make_tfrecords or not os.path.exists(base_dir):
        train_files, test_files = convert_datasets(base_dir)
    else:
        train_files = [base_dir / 'train' / file_name for file_name in os.listdir(base_dir / 'train')]
        test_files = [base_dir / 'test' / file_name for file_name in os.listdir(base_dir / 'test')]
    # TODO get some validation files
    valid_files = []

    # make the datasets
    train_dataset: tf.data.Dataset = get_dataset(train_files, img_shape=data_img_shape)
    valid_dataset: tf.data.Dataset = get_dataset(valid_files, img_shape=data_img_shape)
    test_dataset: tf.data.Dataset = get_dataset(test_files, img_shape=data_img_shape)

    # TODO repeat, shuffle, scale, batch and prefetch the datasets

    # check our dataset by visualizing it. Note this can go forever if you let it
    if visualize_dataset:
        for img_batch, label_batch in train_dataset.as_numpy_iterator():
            plt.imshow(img_batch[0])
            print(label_batch[0])
            plt.show()

    # the cifar dataset comes with these fixed values so hard code them
    total_samples = 5000 * 10
    total_cats = 5000 * 1
    total_not_cats = 5000 * 9
    print(
        f"Total Samples: {total_samples}\nTotal Cats   : {total_cats}\nAccuracy if I always guess not cat: {(total_samples - total_cats) / total_samples * 100}%")

    class_weight = {
        0: 1 / total_not_cats,
        1: 1 / total_cats,
    }

    total_test_samples = 1000 * 10
    total_test_cats = 1000 * 1
    total_test_not_cats = 1000 * 9

    # Make our model using a generator
    saved_model_filename = 'model.h5'
    if not os.path.exists(saved_model_filename) or train_model:
        # TODO put your model from lab 4 here
        model = Model()
        model.save(saved_model_filename)
    else:
        model = load_model(saved_model_filename)

    eval_dataset = valid_dataset if show_metrics_for_valid_dataset else test_dataset
    visualize_model_dataset(model=model,
                            dataset=eval_dataset,
                            eval_samples=total_test_samples)

    # Now for a pre-trained network
    # This is similar to the example in Section 5.3 from the Chollet book
    saved_model_filename = 'model_pretrained.h5'
    if not os.path.exists(saved_model_filename) or train_pre_trained_model:
        # TODO get a conv_base from a pre-trained keras model
        conv_base = None

        # TODO make your pre-trained network using the conv_base
        model_pretrained = Model()
        model_pretrained.save(saved_model_filename)
    else:
        model_pretrained = load_model(saved_model_filename)

    eval_dataset = valid_dataset if show_metrics_for_valid_dataset else test_dataset
    visualize_model_dataset(model=model_pretrained,
                            dataset=eval_dataset,
                            eval_samples=total_test_samples)


if __name__ == "__main__":
    main()
