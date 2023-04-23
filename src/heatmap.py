from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

tf.compat.v1.disable_eager_execution()
# I usually do not like referencing files relative to the running script but this is for just two images
_FILE_PATH = Path(__file__).parent.absolute()


# this is from Chollet Jupyter Notebooks Section 5.4
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/660498db01c0ad1368b9570568d5df473b9dc8dd/first_edition/5.4-visualizing-what-convnets-learn.ipynb
def main():
    # Heat map of class activation
    # The local path to our target image
    # This is similar to the example in Section 5.4 from the Chollet book
    img_names = ['artemis.jpg', 'freya.jpg']
    model = VGG16(weights='imagenet')

    for img_name in img_names:
        img_path = str(_FILE_PATH / img_name)

        # `img` is a PIL image of size 224x224
        img = image.load_img(img_path, target_size=(224, 224))

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x / 255.)

        # TODO: make a prediction for the image
        print(f'Predicted classes for {img_name}: ...')

        # TODO: make a heatmap from the model using the input image

        heatmap: np.ndarray = np.zeros((14, 14))
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap / 255.)

        # We use cv2 to load the original image
        img = cv2.imread(img_path)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img

        # for some reason the plot shows the colors backwards but the saved image does not. weird
        plt.subplot(1, 3, 3)
        # flip the BGR of cv2 to the normal RGB
        plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        # Save the image to disk
        cv2.imwrite(str(_FILE_PATH / f"heatmap_{img_name}"), superimposed_img)

        plt.show()


if __name__ == "__main__":
    main()
