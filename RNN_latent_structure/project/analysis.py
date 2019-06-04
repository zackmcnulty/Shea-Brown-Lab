'''

analysis.py


give it a folder that has a bunch of movie files and an optional labels.csv file. Perform one of the below tests on the files.

'''

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, SimpleRNN, Reshape, ZeroPadding2D, Cropping2D, TimeDistributed, Flatten, ConvLSTM2D
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras import backend as K
import os
import sys
from pathlib import Path
import numpy as np
import cv2 # for reading videos
from matplotlib import pyplot as plt
import argparse
import time
import pandas as pd

# HELPFUL DEBUGGING METHOD
import inspect
def p(x):
    frame = inspect.stack()[1]
    exp = frame.code_context[0].strip()[2:-1]
    print(f"{exp}: {x}")

# note now we are not going to vectorize our data because we care about the local structure
# instead, input will be fed in as matrices (i.e. frame by frame)
parser = argparse.ArgumentParser()
parser.add_argument('-load', help='file name for a previously trained RNN that you wish to train further.', required=True) # specify a full pre-trained RNN model to load
parser.add_argument('-movie_folder', help='path to folder with movie files to perform analysis on',required=True)
args = parser.parse_args()



# Load movie clips ===============================================================================`

# NOTE: Unlike in the convolutional_autoencoder case where we just stacked all the frames to form our 
# inputs, we want to retain this data as sequences so we can do sequence prediction with our RNN (i.e. predict
# the future frames given a set of current frames). Thus,
# our inputs will be stored as full films and the outputs will be staggered in time
# Thus, inputs to neural net will be of the form (# films, # frames/film, frame height, frame width)
# with each single input as (1, # frames, frame height, frame width)

num_movies = 0

# count movies first so I can pre-allocate space in numpy arrays
for i, f in enumerate(os.scandir(args.movie_folder)):

    if '.mp4' in f.name:
        num_movies += 1

        # should be the same for all movies
        if i == 0:  
            cap = cv2.VideoCapture(f.path)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_shape = [frameHeight, frameWidth]
    
# you start with frameCount frames. Since the input and output have to be staggered by dt, this means
# that the input and output can have at most (frameCount - dt) frames as dt of their frames do not overlap.

# LOAD THE OLD dt value from the model name
if 'dt' in args.load:
    model_args = args.load.split('_')
    dt = int(model_args[model_args.index('dt') + 1])
else:
    dt = 0
num_frames = frameCount - dt

movies = np.empty((num_movies, num_frames, frameHeight, frameWidth, 1))
labels  = np.empty((num_movies, 1))

ind = 0
movie_num = 0
for f in os.scandir(args.movie_folder):

    if '.mp4' in f.name:
        cap = cv2.VideoCapture(f.path)
        while ind < num_frames: 
            # ret is a boolean that captures whether or not the frame was read properly (i.e.
            # whether or not we have reached end of video and run out of frames to read)
            ret , frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            movies[movie_num, ind, :, :, 0] = gray / 255.0
            ind += 1
            

        movie_num += 1
        ind = 0

    # load labels from labels.csv file
    elif 'labels' in f.name:
        filename = Path(args.movie_folder) / 'labels.csv'
        df = pd.read_csv(filename, sep=',', header=None)
        labels = df.values



# ================================================================================================

# make predictions on training dataset!

full_model = load_model(args.load)

# Reconstruct the network but leave out the decoding layer; this makes the network output
# the activations of the RNN
rnn = Sequential()
cnn = Sequential()
for layer in full_model.layers:
    rnn.add(layer)
    if layer.name == 'rnn': break
    cnn.add(layer)

# RUN SVD and plot the singular values
if True:
    rnn_predicted = rnn.predict(movies) # the [0] just takes predictions for first video
    rnn_predicted = rnn_predicted.reshape(num_movies * num_frames, -1)
    p(rnn_predicted.shape)

    cnn_predicted = cnn.predict(movies) # the [0] just takes predictions for first video
    cnn_predicted = cnn_predicted.reshape(num_movies * num_frames, -1)
    p(cnn_predicted.shape)

    flattened_movies = movies.reshape(num_movies * num_frames, -1)
    s_movies = np.linalg.svd(flattened_movies, compute_uv=False)


    # PLOT singular values
    plt.figure(76)
    plt.subplot(131)
    plt.title('Movies Singular Values')
    plt.xlabel('Index j')
    plt.ylabel('jth Singular value')
    plt.semilogy(s_movies[:64] / max(s_movies), 'ro')
    plt.ylim([1e-4, 10])

    
    [u_rnn, s_rnn, vh_rnn] = np.linalg.svd(rnn_predicted, full_matrices = False)

    plt.subplot(132)
    plt.title('RNN Neural Representation Singular Values')
    plt.xlabel('Index j')
    plt.ylabel('jth Singular value')
    #plt.semilogy(s_rnn / max(s_rnn), 'ro')
    #plt.ylim([1e-4, 10])
    plt.plot(s_rnn / max(s_rnn), 'ro')
    plt.ylim([-0.1,1.1])


    [u_cnn, s_cnn, vh_cnn] = np.linalg.svd(cnn_predicted, full_matrices = False)

    plt.subplot(133)
    plt.title('CNN Neural Representation Singular Values')
    plt.xlabel('Index j')
    plt.ylabel('jth Singular value')
    #plt.semilogy(s_cnn / max(s_cnn), 'ro')
    #plt.ylim([1e-4, 10])
    plt.plot(s_cnn / max(s_cnn), 'ro')
    plt.ylim([-0.1,1.1])
    plt.show()


# plot stuff ====================================================================================

