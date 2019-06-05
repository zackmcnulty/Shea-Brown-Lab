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
parser.add_argument('--svm', action='store_true')
parser.add_argument('--uniform', action='store_true')
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
labels  = np.empty((num_movies, ))

ind = 0
movie_num = 0

# SORT FILENAMES SO THEY MATCH UP WITH THE LABELS IN labels.cvs; the traversing of directory
# occurs in a random order
file_names = []
for f in os.scandir(args.movie_folder):
    if '.mp4' in f.name:
        file_names.append(f.path)
    elif 'labels' in f.name:
        filename = Path(args.movie_folder) / 'labels.csv'
        df = pd.read_csv(filename, sep=',', header=None)
        labels = df.values
        labels = np.reshape(labels, (-1, ))

file_names = sorted(file_names)

for f in file_names:
    cap = cv2.VideoCapture(f)
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

# calculate activations of RNN and encoder in response to given movies
rnn_representation = rnn.predict(movies) 
cnn_representation = cnn.predict(movies) 
p(rnn_representation.shape)
p(cnn_representation.shape)

# RUN SVD and plot the singular values
if False:
    rnn_rep = rnn_representation.reshape(num_movies * num_frames, -1)
    p(rnn_representation.shape)

    cnn_rep = cnn_representation.reshape(num_movies * num_frames, -1)
    p(cnn_representation.shape)

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

    
    [u_rnn, s_rnn, vh_rnn] = np.linalg.svd(rnn_rep, full_matrices = False)

    plt.subplot(132)
    plt.title('RNN Neural Representation Singular Values')
    plt.xlabel('Index j')
    plt.ylabel('jth Singular value')
    #plt.semilogy(s_rnn / max(s_rnn), 'ro')
    #plt.ylim([1e-4, 10])
    plt.plot(s_rnn / max(s_rnn), 'ro')
    plt.ylim([-0.1,1.1])


    [u_cnn, s_cnn, vh_cnn] = np.linalg.svd(cnn_rep, full_matrices = False)

    plt.subplot(133)
    plt.title('CNN Neural Representation Singular Values')
    plt.xlabel('Index j')
    plt.ylabel('jth Singular value')
    #plt.semilogy(s_cnn / max(s_cnn), 'ro')
    #plt.ylim([1e-4, 10])
    plt.plot(s_cnn / max(s_cnn), 'ro')
    plt.ylim([-0.1,1.1])
    plt.show()



# RUN CLASSIFICATION
if args.svm:
    from sklearn import svm
    classifier_rnn = svm.SVC(kernel='linear')
    classifier_cnn = svm.SVC(kernel='linear')

    # NOTE: don't need a test set?

    # NOTE: randomly choose a frame in movie for classification?
    rnn_random_frames = np.empty((num_movies, rnn_representation.shape[2]))
    cnn_random_frames = np.empty((num_movies, rnn_representation.shape[2]))

    random_indices = np.random.choice(list(range(1, num_frames)), num_movies)

    for movie_num in range(num_movies):
        rnn_random_frames[movie_num, :] = rnn_representation[movie_num, random_indices[movie_num], :]
        cnn_random_frames[movie_num, :] = cnn_representation[movie_num, random_indices[movie_num], :]

    print(rnn_random_frames.shape)

    classifier_rnn.fit(rnn_random_frames, labels)
    p(classifier_rnn.score(rnn_random_frames, labels)) # print RNN classification score
    
    classifier_cnn.fit(cnn_random_frames, labels)
    p(classifier_cnn.score(cnn_random_frames, labels)) # print CNN classification score

    # NOTE Do we really want to stack all frames for a given movie? Or should we
    # do it frame by frame instead? or maybe just use the representation for the last
    # frame in classification?
    rnn_representation = rnn_representation.reshape(num_movies, -1)
    classifier_rnn.fit(rnn_representation, labels)
    p(classifier_rnn.score(rnn_representation, labels))


    cnn_representation = cnn_representation.reshape(num_movies, -1)
    classifier_cnn.fit(cnn_representation, labels)
    p(classifier_cnn.score(cnn_representation, labels))


    # CONTROL: randomly assign labels to representations

if args.uniform:

    # CNN interesting neurons: 2,6 = flat,  32=periodic, 16=two-peak periodic
    # RNN interesting neurons: 25=periodic, 61
    #neuron_numbers = [1,2,3,4,5,6,7,8] # which neurons to make plots for
    neuron_numbers = list(range(61, 64)) # which neurons to make plots for

    # PLOT activation of a single neuron over course of an entire movie
    plot_rows = 1
    plot_cols = len(neuron_numbers)

    plt.figure(99)
    plt.xlabel('Frame number')
    plt.ylabel('Neuron Activation')
    movie_nums = list(range(180))
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('RNN representation (Neuron {})'.format(neuron))

        for num in movie_nums:
            plt.plot(rnn_representation[num, :, neuron])

#        plt.legend(['Movie {} ({} degrees)'.format(num + 1, int(labels[num])) for num in movie_nums])

    plt.show()

    plt.figure(123)
    plt.xlabel('Frame number')
    plt.ylabel('Neuron Activation')
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('Decoder representation (Neuron {})'.format(neuron))

        for num in movie_nums:
            plt.plot(cnn_representation[num, :, neuron])

#        plt.legend(['Movie {} ({} degrees)'.format(num + 1, int(labels[num])) for num in movie_nums])

    plt.show()



    # PLOT activation of a single neuron at a specific frame across many different movies (with different angles of oscillation)
    # RNN important: 44=clear angle preference, 58
    # CNN important: 44=clear angle preference, 58

    plot_rows = 1
    plot_cols = len(neuron_numbers) 

    plt.figure(99)
    plt.xlabel('Axis Angle (degrees)')
    plt.ylabel('Neuron Activation')
    frame_nums = list(range(30))
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('RNN representation (Neuron {})'.format(neuron))

        for num in frame_nums:
            plt.plot(labels, rnn_representation[:, num, neuron])

#        plt.legend(['Frame {} )'.format(num + 1) for num in movie_nums])

    plt.show()

    plt.figure(123)
    plt.xlabel('Frame number')
    plt.ylabel('Neuron Activation')
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('Decoder representation (Neuron {})'.format(neuron))

        for num in frame_nums:
            plt.plot(labels, cnn_representation[:, num, neuron])

#        plt.legend(['Frame {}'.format(num + 1) for num in movie_nums])

    plt.show()



    # PLOT activation of a single neuron at a specific frame across many different movies (with different angles of oscillation)


