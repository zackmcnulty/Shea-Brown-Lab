'''
Video prediction using PredNet
'''

import os
import sys

sys.path.append('./prednet') # append prednet folder to path so we can import prednet layer

import prednet # import the custom prednet layer, which also import the relevant keras, tensorflow, etc that it requires.

from keras.layers import ZeroPadding2D, Cropping2D, TimeDistributed
from keras.models import Model, Sequential, load_model
from keras import regularizers as reg


from pathlib import Path
import numpy as np
import cv2 # for reading videos
from matplotlib import pyplot as plt
import argparse
import time

# note now we are not going to vectorize our data because we care about the local structure
# instead, input will be fed in as matrices (i.e. frame by frame)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train for')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for SGD in training')
parser.add_argument('--name', help='name of model file')
parser.add_argument('--load', help='load a keras model file of a previously trained model') # specify a file to load
parser.add_argument('--save_fig', action='store_true', default=False, help='save figure of training results') 
parser.add_argument('--show_movie', action='store_true', default=False, help='show movie of training results') 
parser.add_argument('--dt', default=1, type=int, help='number of frames ahead to predict in the movie')

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_name = args.name
num_results_shown = 10 # number of reconstructed frames vs original to show on test set

# Save the Keras model
if args.load is not None:

    if args.dt is None:
        name = args.load[:-3]
        model_args = name.split('_')
        args.dt = int(model_args[model_args.index('dt') + 1]) 

    if args.name is None: 
        model_filename = Path(args.load)
    else: 
        model_filename = Path(model_name + "_dt_"  + str(args.dt) + ".h5")


else:
    if args.name is None: model_name = 'rnn_predictor_prednet'
    model_filename = Path(model_name + "_dt_"  + str(args.dt) + ".h5")
    model_filename = 'models' / model_filename 

if model_filename.exists() and epochs > 0:
    answer = input('File name ' + model_filename.name + ' already exists. Overwrite [y/n]? ')
    if not 'y' in answer.lower():
        sys.exit()

# Load movie clips ===============================================================================`

num_train_movies = 0
num_test_movies = 0

# count movies first so I can pre-allocate space in numpy arrays
for i, f in enumerate(os.scandir('./movie_files')):
    if f.name.startswith('train'):
        num_train_movies += 1
    elif f.name.startswith('test'):
        num_test_movies += 1

    # should be the same for all movies
    if i == 0:  
        cap = cv2.VideoCapture(f.path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_shape = [frameHeight, frameWidth]

# you start with frameCount frames. Since the input and output have to be staggered by dt, this means
# that the input and output can have at most (frameCount - dt) frames as dt of their frames do not overlap.
num_frames = frameCount - args.dt

x_train = np.empty((num_train_movies, num_frames, frameHeight, frameWidth, 1))
y_train = np.empty((num_train_movies, num_frames, frameHeight, frameWidth, 1))
x_test = np.empty((num_test_movies, num_frames, frameHeight, frameWidth, 1))
y_test = np.empty((num_test_movies, num_frames, frameHeight, frameWidth, 1))

train_ind = 0
train_movie_num = 0
test_ind = 0
test_movie_num = 0

for f in os.scandir('./movie_files'):
    cap = cv2.VideoCapture(f.path)
    while True:
        # ret is a boolean that captures whether or not the frame was read properly (i.e.
        # whether or not we have reached end of video and run out of frames to read)
        ret , frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if f.name.startswith('train'):
            if train_ind < num_frames:
                x_train[train_movie_num, train_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            if train_ind >= args.dt:
                y_train[train_movie_num, train_ind - args.dt, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]

            train_ind += 1

        elif f.name.startswith('test'):
            if test_ind < num_frames:
                x_test[test_movie_num, test_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            if test_ind >= args.dt:
                y_test[test_movie_num, test_ind - args.dt, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]

            test_ind += 1

    if f.name.startswith('train'):
        train_movie_num += 1
        train_ind = 0

    elif f.name.startswith('test'):
        test_movie_num += 1
        test_ind =  0




# convert the image into a lower dimensional representation

# Conv2D( depth, kernel/filter size, ...)
# The depth refers to the number of neurons connected to the same receptive field in the input.
# In the case below, we have 16 neurons connected to each 3 x 3 receptive field of pixels in 
# the input. These receptive fields are generated by moving the 3 x 3 filter one pixel at a time 
# (there is another parameter called 'strides' that can specify how many pixels to move each time).
# These 16 neurons convolve with their receptive field to capture the local patterns. Important
# in cases where local structure is the most important. As these layers only need to consider their
# receptive field, they are NOT connected to any other neurons in the previous layer

# To the right, we have shown how the dimensionality of the input changes. 

# NOTE: maxpooling does NOT effect the depth of the layers (number of filters); it only reduces the number
# of receptive fields.

# Maxpooling reduces the dimensionality


#Define Neural Network Structure ===============================================================================`
if args.load is not None:
    # do stuff
    model = load_model(args.load, {"PredNet": prednet.PredNet})
else:
    image_shape.append(1)

    model = Sequential()

    # the image size has to be divisible by 2^(nb of layers - 1) because of the cyclical 2x2 max-pooling and upsampling operations
    # Here, we just pad the image with zeros to accomplish this.
    model.add(TimeDistributed(ZeroPadding2D(((2,1),(2,1)), input_shape = image_shape)))  # (61,61, 1) --> (64,64, 1)

    # PredNet: see their GitHub for description of all parameters (https://github.com/coxlab/prednet/blob/master/prednet.py)
    stack_sizes = (1, 16, 32)
    R_stack_sizes = stack_sizes
    A_filt_sizes=(3,3) # size of convolutional windows of A units
    Ahat_filt_sizes=(3,3,3) # size of convolutional windows of Ahat units
    R_filt_sizes=(3,3,3) # size of convolutional windows for R units; the recurrent LSTM units (all LSTM layers have same filter size)

    # extrapolation start time to force multistep prediction
    if args.dt == 1:
        start_time = None
    else:
        start_time = num_frames - args.dt

    model.add(prednet.PredNet(
                                stack_sizes=stack_sizes,
                                R_stack_sizes=R_stack_sizes,
                                A_filt_sizes=A_filt_sizes,
                                Ahat_filt_sizes=Ahat_filt_sizes,
                                R_filt_sizes=R_filt_sizes,
                                return_sequences=True, 
                                pixel_max=1,

                                extrap_start_time = start_time, # forcing multi-step prediction

                                output_mode='prediction' # return predicted videos
    
    ))

    model.add(TimeDistributed(Cropping2D(((2,1),(2,1))))) # (64,64,1) -> (61,61, 1)

    model.compile(optimizer = 'adam', loss='binary_crossentropy')


# ================================================================================================
# fit model (train network)!
if epochs > 0:
    model.fit(x_train, x_train,
              epochs = epochs, 
              batch_size = batch_size, 
              shuffle = True,
              validation_data = (x_test, x_test))

    # NOTE: Saves the model to the given model name in the folder ./models
    model.save(str(model_filename))

model.summary()









# make predictions on training dataset!
predicted_images_train = model.predict(x_train[:1, :, :, :, :])[0] # the [0] just takes predictions for first video
true_images_train = y_train[0, :, :, :, :]
initial_images_train = x_train[0, :, :, :, :]

# plot stuff ====================================================================================

n = num_results_shown # number of frames to display

plt.figure(figsize=(20,4))
plt.title('Training dataset')
for i in range(n):
    # display original at time t (in top row)
    ax = plt.subplot(3, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(initial_images_train[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t)')

    # display original at time t+dt (in middle row)
    ax = plt.subplot(3, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(true_images_train[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t+dt)')

    # display predicted at time t+dt (in bottom row)
    ax = plt.subplot(3, n, i+ 1 + 2*n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images_train[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Predicted (t+dt)')

plt.show()











# make predictions on test dataset!
predicted_images_test = model.predict(x_test[5:6, :, :, :, :])[0] # the [0] just takes predictions for first video
true_images_test = y_test[0, :, :, :, :]
initial_images_test = x_test[0, :, :, :, :]


# plot stuff ====================================================================================

n = num_results_shown # number of frames to display

plt.figure(figsize=(20,4))
plt.title("testing dataset")
for i in range(n):
    # display original at time t (in top row)
    ax = plt.subplot(3, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(initial_images_test[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t)')

    # display original at time t+dt (in middle row)
    ax = plt.subplot(3, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(true_images_test[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t+dt)')

    # display predicted at time t+dt (in bottom row)
    ax = plt.subplot(3, n, i+ 1 + 2*n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images_test[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Predicted (t+dt)')

plt.show()

if args.save_fig:
    plt.savefig('analysis_plots/prednet/{}.png'.format(model_filename.name[:-3]))


# ========== make movie of results! ===============================

if args.show_movie:
    n = 50
    plt.ion()
    plt.figure(figsize=(20,4))
    plt.show()

    # make results for training dataset
    for i in range(n):
        # display original at time t (in top row)
        ax = plt.subplot(3, 1, 1) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(initial_images_train[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        # display original at time t+dt (in middle row)
        ax = plt.subplot(3, 1, 2) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(true_images_train[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        # display predicted at time t+dt (in bottom row)
        ax = plt.subplot(3, 1, 3) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(predicted_images_train[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        plt.draw()
        plt.pause(0.001)
        plt.clf()

    # make results for testing dataset
    for i in range(n):
        # display original at time t (in top row)
        ax = plt.subplot(3, 1, 1) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(initial_images_test[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        # display original at time t+dt (in middle row)
        ax = plt.subplot(3, 1, 2) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(true_images_test[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        # display predicted at time t+dt (in bottom row)
        ax = plt.subplot(3, 1, 3) # which subplot to work with; 2 rows, n columns, slot i+1
        plt.imshow(predicted_images_test[i].reshape(image_shape[0], image_shape[1]))
        plt.gray()
        ax.get_xaxis().set_visible = False
        ax.get_yaxis().set_visible = False

        plt.draw()
        plt.pause(0.001)
        plt.clf()
