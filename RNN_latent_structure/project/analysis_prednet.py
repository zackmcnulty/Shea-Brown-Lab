'''

analysis.py


give it a folder that has a bunch of movie files and an optional labels.csv file. Perform one of the below tests on the files.

'''

import os
import sys
from pathlib import Path
import numpy as np
import cv2 # for reading videos
from matplotlib import pyplot as plt
import argparse
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import dynamical_systems as ds
import get_trajectory
import re


from sklearn.decomposition import PCA

sys.path.append('./prednet') # append prednet folder to path so we can import prednet layer

import prednet # import the custom prednet layer, which also import the relevant keras, tensorflow, etc that it requires.

import keras
from keras.layers import ZeroPadding2D, Cropping2D, TimeDistributed
from keras.models import Model, Sequential, load_model, clone_model
from keras import regularizers as reg 
from keras import backend as K



# HELPFUL DEBUGGING METHOD
import inspect
def p(x):
    frame = inspect.stack()[1]
    exp = frame.code_context[0].strip()[2:-1]
    print(f"{exp}: {x}")

# note now we are not going to vectorize our data because we care about the local structure
# instead, input will be fed in as matrices (i.e. frame by frame)
parser = argparse.ArgumentParser()
parser.add_argument('--load', help='file name for a previously trained RNN that you wish to train further.', required=True) # specify a full pre-trained RNN model to load
parser.add_argument('--folder', help='path to folder with movie files to perform analysis on',required=True)
parser.add_argument('--make_figs', action='store_true', default=False, help="shows the neural representations plotted on the Principal Components and color coded by feature")
parser.add_argument('--no_demean', action='store_true', default=False,  help='flag to not demean the rnn/cnn neuron representations')
args = parser.parse_args()

model_name = args.load[7:-3]



# Load movie clips ===============================================================================`

# NOTE: Delete when done testing
max_movies = 360
num_movies = 0

# count movies first so I can pre-allocate space in numpy arrays
for i, f in enumerate(os.scandir(args.folder)):

    if '.mp4' in f.name:
        num_movies += 1

        # should be the same for all movies
        if num_movies == 1:
            cap = cv2.VideoCapture(f.path)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_shape = [frameHeight, frameWidth]
    
# you start with frameCount frames. Since the input and output have to be staggered by dt, this means
# that the input and output can have at most (frameCount - dt) frames as dt of their frames do not overlap.

# LOAD THE OLD dt value from the model name
#if 'dt' in args.load:
model_args = args.load[:-3].split('_')
dt = int(model_args[model_args.index('dt') + 1])
#else:
#    dt = 0

num_frames = frameCount - dt

# NOTE: change this after testing
#movies = np.empty((num_movies, num_frames, frameHeight, frameWidth, 1))
movies = np.empty((max_movies, num_frames, frameHeight, frameWidth, 1))

ind = 0
movie_num = 0

# SORT FILENAMES SO THEY MATCH UP WITH THE LABELS IN labels.cvs; the traversing of directory
# occurs in a random order
file_names = []
for f in os.scandir(args.folder):
    if '.mp4' in f.name:
        file_names.append(f.path)
    elif 'labels' in f.name:
        filename = Path(args.folder) / 'labels.csv'
        df = pd.read_csv(filename, sep=',', header=0)
        print(df.columns)
        labels = df.values
        #labels = np.reshape(labels, (-1, ))

# Sort filenames so they line up appropriately with the labels which are ordered by number
file_names = sorted(file_names)

for f in file_names:

    # NOTE: remove me after testing
    if movie_num >= max_movies: break

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

# NOTE: remove after testing
labels = labels[:max_movies, :]
print(labels.shape)
print(labels)
print(movies.shape)


# ================================================================================================

output_mode = 'R2'
digit_location = re.search("\d", output_mode).start()

# make predictions on training dataset!

full_model = load_model(args.load, {"PredNet": prednet.PredNet})
full_model.pop()

model = keras.models.clone_model(full_model)
model.layers[-1].output_mode = output_mode
model.layers[-1].output_layer_type = output_mode[:digit_location]
model.layers[-1].output_layer_num = int(output_mode[digit_location:])


model.set_weights(full_model.get_weights())

model.build(full_model.input_shape)
print(model.summary())


# (movie number, frame number, neuron coordinates...)
neural_rep = model.predict(movies)
neural_rep = neural_rep.reshape(neural_rep.shape[0], -1)
print(neural_rep.shape)

if not args.no_demean:
    mean_activity = np.mean(np.mean(neural_rep, axis=0), axis=0)
    neural_rep -= mean_activity


# TODO: Adjust all this to fit the PREDNET architecture ==============================================

# PCA ================================================================================================

pca = PCA(n_components=5)
pca.fit(neural_rep)

pca_coordinates = pca.transform(neural_rep)

print(pca_coordinates.shape)


if args.make_figs:
    for feature in range(1, labels.shape[1]):
        # Choose some way to compare the datapoints. Adjusting the color is good for continuous variables, while adusting the marker
        # is more suitable for categorical variables
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = labels[:, feature] # x coord
        cmap = 'hot'
        markers = None

        xs = pca_coordinates[:, 0]
        ys = pca_coordinates[:, 1]
        zs = pca_coordinates[:, 2]

        ax.scatter(xs, ys, zs, c=colors, marker=markers, cmap=cmap)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        plt.title(df.columns[feature])

        plt.savefig('analysis_plots/PCA/PC123{}_{}_{}.png'.format(model_name, output_mode, df.columns[feature][1:]))
        plt.show()





    for feature in range(1, labels.shape[1]):
        # Choose some way to compare the datapoints. Adjusting the color is good for continuous variables, while adusting the marker
        # is more suitable for categorical variables
        plt.figure()

        colors = labels[:, feature] # x coord
        cmap = 'hot'
        markers = None

        xs = pca_coordinates[:, 3]
        ys = pca_coordinates[:, 4]

        plt.scatter(xs,ys,c=colors,marker=markers,cmap=cmap)
        plt.xlabel('PC 4')
        plt.ylabel('PC 5')

        plt.title(df.columns[feature])

        plt.savefig('analysis_plots/PCA/PC45{}_{}_{}.png'.format(model_name, output_mode, df.columns[feature][1:]))
        plt.show()



# =======================================================================================================

number_coords = 3
coordinates = pca_coordinates[:, 0:number_coords] 


# https://stackoverflow.com/questions/48312205/find-the-k-nearest-neighbours-of-a-point-in-3d-space-with-python-numpy
from scipy.spatial import distance
D = distance.squareform(distance.pdist(coordinates)) # distance matrix
closest = np.argsort(D, axis=1)
print(closest)

k = 9 # For each point, find the 3 closest points
#print(closest[:, 1:k+1])

np.savetxt(args.folder + "/closest.csv", closest, delimiter=',')




"""
for i in range(closest.shape[0]):

    plt.ion()
    plt.figure(num=i+1, figsize=(20,4))
    plt.show()

    # make results for training dataset
    for frame in range(num_frames):

        for neighbor in range(k):

            next_movie = closest[i, neighbor]
            # display original at time t (in top row)
            ax = plt.subplot(np.ceil(np.sqrt(k)), np.ceil(np.sqrt(k)), neighbor + 1) # which subplot to work with; 2 rows, n columns, slot i+1
            #ax = plt.subplot(k, 1, neighbor + 1) # which subplot to work with; 2 rows, n columns, slot i+1

            plt.imshow(movies[next_movie, frame, :, :, :].reshape(frameHeight, frameWidth))
            plt.gray()
            ax.get_xaxis().set_visible = False
            ax.get_yaxis().set_visible = False



        plt.draw()
        plt.clf()

    plt.close()
"""
