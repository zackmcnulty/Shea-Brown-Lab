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
from pydmd import DMD
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm



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
parser.add_argument('--svd', action='store_true')
parser.add_argument('--positional_activity', action='store_true')
parser.add_argument('--uniform', action='store_true')
parser.add_argument('--pca', action='store_true')
parser.add_argument('--isomap', type=int)
parser.add_argument('--no_demean', action='store_true', default=False,  help='flag to not demean the rnn/cnn neuron representations')
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

# Sort filenames so they line up appropriately with the labels which are ordered by number
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


# Mean subtract from data
if not args.no_demean:
    rnn_mean = np.mean(np.mean(rnn_representation, axis=0), axis=0)
    cnn_mean = np.mean(np.mean(cnn_representation, axis=0), axis=0)
    rnn_representation -= rnn_mean
    cnn_representation -= cnn_mean

#p(cnn_mean.shape)
#p(rnn_mean.shape)

#p(rnn_representation.shape)
#p(cnn_representation.shape)

# ========================================================================================================================
# RUN SVD and plot the singular values
if args.svd:

    # Here, we store the neural representation of  each frame as a column in our matrix. Thus, when we
    # take the SVD the columns of U will form a basis for neural representation space and the columns of SV*
    # give the coordinates within this space.
    rnn_rep = rnn_representation.reshape(-1, num_movies * num_frames)
    p(rnn_representation.shape)

    cnn_rep = cnn_representation.reshape(-1, num_movies * num_frames)
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



    # RNN trajectories in PC space over time (frame by frame)
    movie_nums = [0]
    num_pcs = 3
    for movie in movie_nums:
        movie_pc_coords = np.zeros((num_frames, 3))
        for frame in range(num_frames):
            movie_pc_coords[frame, :num_pcs] =  np.multiply(s_rnn[:num_pcs], vh_rnn[:num_pcs, movie + frame])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = cm.rainbow(np.linspace(0, 1, num_frames))
        for i in range(num_frames):
            ax.plot([movie_pc_coords[i, 0]], [movie_pc_coords[i, 1]], [movie_pc_coords[i,2]], 'o', color=colors[i], markersize=5)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('RNN PC Trajectory (movie {})'.format(movie))
        plt.show()

    # CNN trajectories in PC space over time (frame by frame)
    movie_nums = [0,5,10,15,20,25,30]
    num_pcs = 3
    for movie in movie_nums:
        movie_pc_coords = np.zeros((num_frames, 3))
        for frame in range(num_frames):
            movie_pc_coords[frame, :num_pcs] =  np.multiply(s_cnn[:num_pcs], vh_cnn[:num_pcs, movie + frame])



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = cm.rainbow(np.linspace(0, 1, num_frames))
        for i in range(num_frames):
            ax.plot([movie_pc_coords[i, 0]], [movie_pc_coords[i, 1]], [movie_pc_coords[i,2]], 'o', color=colors[i], markersize=5)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('CNN PC Trajectory (movie {})'.format(movie))
        plt.show()

# ========================================================================================================================





# ========================================================================================================================


if args.uniform:

    # CNN interesting neurons: 2,6 = flat,  32=periodic, 16=two-peak periodic
    # RNN interesting neurons: 25=periodic, 61
    #neuron_numbers = [16,25, 32] # which neurons to make plots for
    neuron_numbers = list(range(10,20))

    # PLOT activation of a single neuron over course of an entire movie
    plot_rows = 2
    plot_cols = len(neuron_numbers) / plot_rows

    plt.figure(99)
    movie_nums = list(range(180))
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('RNN representation (Neuron {})'.format(neuron))
        plt.xlabel('Frame number')
        plt.ylabel('Neuron Activation')

        for num in movie_nums:
            plt.plot(rnn_representation[num, :, neuron])

#        plt.legend(['Movie {} ({} degrees)'.format(num + 1, int(labels[num])) for num in movie_nums])

    #plt.show()

    plt.figure(123)
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('Decoder (Neuron {})'.format(neuron))
        plt.xlabel('Frame number')
        plt.ylabel('Neuron Activation')

        for num in movie_nums:
            plt.plot(cnn_representation[num, :, neuron])

#        plt.legend(['Movie {} ({} degrees)'.format(num + 1, int(labels[num])) for num in movie_nums])

    plt.show()



    # PLOT Principal Components of time dynamics
    num_pcs = 1

    plt.figure(146)
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('RNN representation (Neuron {})'.format(neuron))
        plt.xlabel('Frame number')
        plt.ylabel('Neuron Activation')
        
#        [u, s, v] = np.linalg.svd(rnn_representation[:, :, neuron].T, full_matrices=False)
        dmd = DMD(svd_rank = num_pcs)
        dmd.fit(rnn_representation[:, :, neuron].T)


        for mode in dmd.modes.T:
#            plt.plot(u[i, :])
            plt.plot(mode.real)
             

        plt.legend(['Mode {}'.format(k+1) for k in range(num_pcs)])
        
    plt.show()


    plt.figure(1445)
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('CNN representation (Neuron {})'.format(neuron))
        plt.xlabel('Frame number')
        plt.ylabel('Neuron Activation')
        
#        [u, s, v] = np.linalg.svd(rnn_representation[:, :, neuron].T, full_matrices=False)

        # DMD fails on the case where CNN activations always zero
        try:
            dmd = DMD(svd_rank = num_pcs)
            dmd.fit(cnn_representation[:, :, neuron].T)


            for mode in dmd.modes.T:
    #            plt.plot(u[i, :])
                plt.plot(mode.real)
        except:
            pass
             

        plt.legend(['Mode {}'.format(k+1) for k in range(num_pcs)])
        
    plt.show()




    # PLOT PRINCIPAL COMPONENTS OF ALL TIME DYNAMICS TOGETHER =======================
    num_pcs = 3
    dmd = DMD(svd_rank = num_pcs)
    dmd.fit(rnn_representation.reshape(64 * num_movies, num_frames).T)

    plt.figure(1201)
    plt.title('All RNN Neurons Modes')
    for mode in dmd.modes.T:
        plt.plot(mode.real)
    
    plt.xlabel('Frame')
    plt.legend(['Mode {}'.format(i+1) for i in range(num_pcs)])
    plt.show()


    dmd = DMD(svd_rank = num_pcs)
    dmd.fit(cnn_representation.reshape(64 * num_movies, num_frames).T)

    plt.figure(1201)
    plt.title('All Encoder Neurons Modes')
    for mode in dmd.modes.T:
        plt.plot(mode.real)

    plt.xlabel('Frame')
    plt.legend(['Mode {}'.format(i+1) for i in range(num_pcs)])
    plt.show()





    # PLOT activation of a single neuron at a specific frame across many different movies (with different angles of oscillation)
    # RNN important: 44=clear angle preference, 58, 14 maybe
    # CNN important: 44=clear angle preference, 58, 27 maybe
    neuron_numbers = [27, 44, 58] # which neurons to make plots for

    plot_rows = 1
    plot_cols = len(neuron_numbers) 

    plt.figure(99)
    frame_nums = list(range(30))
    for i, neuron in enumerate(neuron_numbers):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.title('RNN representation (Neuron {})'.format(neuron))
        plt.xlabel('Axis Angle (degrees)')
        plt.ylabel('Neuron Activation')

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
        plt.xlabel('Axis Angle (degrees)')
        plt.ylabel('Neuron Activation')

        for num in frame_nums:
            plt.plot(labels, cnn_representation[:, num, neuron])

#        plt.legend(['Frame {}'.format(num + 1) for num in movie_nums])

    plt.show()




# =========================================================================================================

# Plot activity with respect to position
if args.positional_activity:

    import dynamical_systems as ds
    import get_trajectory

    # RNN Positional Activities
    #neuron_nums=list(range(64))
    neuron_nums=[]
    for neuron in neuron_nums:

        plt.figure()
        cmap = plt.get_cmap('plasma')
        
        for i in range(num_movies):
            # First entry in all_parameter_vals is the parameter number: 
            next_theta = np.pi * float(labels[i]) / 180.0
            objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=next_theta), )
            (t_vals, all_parameter_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames, return_all=True, num_params=3)

            for j in range(num_frames):
                rnn_activity = rnn_representation[i, j, neuron]
                plt.plot([all_parameter_vals[0, 0, j]], [all_parameter_vals[1,0, j]], 'o', markersize=5, color=cmap(rnn_activity))


        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title('RNN Neuron {}'.format(neuron + 1))
        plt.savefig('analysis_plots/positional_activity/rnn_neuron_{:02}.png'.format(neuron + 1))
        #plt.show()



    # CNN Positional Activities
    neuron_nums = []
    for neuron in neuron_nums:

        plt.figure()
        cmap = plt.get_cmap('plasma')
        
        for i in range(num_movies):
            # First entry in all_parameter_vals is the parameter number: 
            next_theta = np.pi * float(labels[i]) / 180.0
            objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=next_theta), )
            (t_vals, all_parameter_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames, return_all=True, num_params=3)

            for j in range(num_frames):
                cnn_activity = cnn_representation[i, j, neuron]
                plt.plot([all_parameter_vals[0, 0, j]], [all_parameter_vals[1,0, j]], 'o', markersize=5, color=cmap(cnn_activity))


        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title('CNN Neuron {}'.format(neuron + 1))
        plt.savefig('analysis_plots/positional_activity/cnn_neuron_{:02}.png'.format(neuron + 1))
#        plt.show()


    # RNN radial Activities
    neuron_nums=list(range(64))
    for neuron in neuron_nums:

        plt.figure()
        cmap = plt.get_cmap('plasma')
        
        for i in range(num_movies):
            # First entry in all_parameter_vals is the parameter number: 
            next_theta = np.pi * float(labels[i]) / 180.0
            objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=next_theta), )
            (t_vals, all_parameter_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames, return_all=True, num_params=3)

            for j in range(num_frames):
                rnn_activity = rnn_representation[i, j, neuron]
                x = all_parameter_vals[0,0, j]
                y = all_parameter_vals[1,0, j]

                center_x = 0.5 # what x,y coordinate the oscillation is centered around
                center_y = 0.5
                r = np.sqrt((x-center_x)**2 + (y-center_y)**2)
                theta = np.arctan2(y - center_y, x - center_x)
                plt.plot([theta], [r], 'o', markersize=5, color=cmap(rnn_activity))


        plt.ylabel('radius')
        plt.xlabel('theta')
        plt.title('RNN Neuron {}'.format(neuron + 1))
        plt.savefig('analysis_plots/positional_activity/rnn_radial_neuron_{:02}.png'.format(neuron + 1))
#        plt.show()

    # CNN radial Activities
    neuron_nums=list(range(64))
    for neuron in neuron_nums:

        plt.figure()
        cmap = plt.get_cmap('plasma')
        
        for i in range(num_movies):
            # First entry in all_parameter_vals is the parameter number: 
            next_theta = np.pi * float(labels[i]) / 180.0
            objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=next_theta), )
            (t_vals, all_parameter_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames, return_all=True, num_params=3)

            for j in range(num_frames):
                cnn_activity = cnn_representation[i, j, neuron]
                x = all_parameter_vals[0,0, j]
                y = all_parameter_vals[1,0, j]

                center_x = 0.5 # what x,y coordinate the oscillation is centered around
                center_y = 0.5
                r = np.sqrt((x-center_x)**2 + (y-center_y)**2)
                theta = np.arctan2(y - center_y, x - center_x)
                plt.plot([theta], [r], 'o', markersize=5, color=cmap(cnn_activity))


        plt.xlabel('theta')
        plt.ylabel('radius')
        plt.title('CNN Neuron {}'.format(neuron + 1))
        plt.savefig('analysis_plots/positional_activity/cnn_radial_neuron_{:02}.png'.format(neuron + 1))
#        plt.show()
