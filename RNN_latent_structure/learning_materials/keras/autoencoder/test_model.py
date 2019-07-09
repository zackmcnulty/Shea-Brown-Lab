'''
test_model.py

This code is for testing models that have been saved as .h5 files on the MNIST database (rather than
retraining the whole model. Pass it a single parameter which is the model name:

usage:

    python3 test_model.py model_name
'''


import sys
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the specified model
try:
    model = load_model(sys.argv[1])
except:
    try:
        path = 'models' / Path(sys.argv[1])
        model = load_model(str(path))
    except:
        raise ValueError('Model not found. Please specify a model filename')


# again, MNIST gives (image, label) pairs, but we dont care about the label as we are just reconstructing
# the image. So throw out the labels with _
(x_train, _), (x_test, _) = mnist.load_data()

# normalize pixels to be in [0,1]
x_test = x_test.astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28,28,1))


predicted_images = model.predict(x_test)



n = 10 # number of digits to display

plt.figure(figsize=(20,4))
for i in range(n):

    # display original (in top row)
    ax = plt.subplot(2, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False

    # display predicted (in bottom row)
    ax = plt.subplot(2, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False

plt.show()

# =======================================================================================
# TODO: analyze stuff about the model
