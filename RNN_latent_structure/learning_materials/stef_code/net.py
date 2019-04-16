## IMPORT LIBRARIES 
import sys
sys.path.append('/home/reca/Public/keras/')
import numpy as np
import scipy.io
import math
import argparse
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam
from keras.layers import SimpleRNN, ActivityRegularization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1
from keras import initializers
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, LambdaCallback, ReduceLROnPlateau, History
import keras.backend as K

# fix random seed for reproducibility
np.random.seed(0)


## DEFINE FUNCTIONS

# function to get activations
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

# function to create one-hot representation for the actions
# matrix with ones and zeros = orthogonal
def onehot(x):
	x_unique = np.unique(x)
	y = np.zeros((x.shape[0],x_unique.shape[0]))
	for x_el, idx in enumerate(x_unique): y[np.where(x == x_el), int(idx)] = 1
	return y

# function to create batches of size N_batch and stacked N_samples times. Total size is (N_batch*N_samples, N_traj, x.shape[-1]=N_x)
def batch(x, N_batch, N_traj, N_samples):
	x_batched = np.zeros((N_samples*N_batch, N_traj, x.shape[-1]))
	N_tot = x.shape[0]
	idxs = 0
	for i_sample in range(0, N_samples*N_batch):
		for i_traj in range(0, N_traj):
			x_batched[i_sample:(i_sample+1), i_traj, :] = x[idxs,:]
			idxs = (idxs + 1)%N_tot    #instead of doing padding it cycles over the same data. So N_batch*N_sample should be N
	return x_batched

# a more efficient version of the function batch 
def batch_time(x,batch):
	x_seq=[]
	for i in range(0,x.shape[-1]):
		x_seq.append(x[:,i].reshape(-1,batch))
	x_seq=np.array(x_seq)
	x_seq=np.rollaxis(x_seq, 0, x_seq.ndim)
	return x_seq

def unbatch_time(x_seq):
	x=x_seq.reshape(-1,x_seq.shape[-1]) 
	return x


## PARSE PARAMETERS AND DATA

# get parser parameters

# allow you to pass in parameters through command line with keywords (i.e. --epochs 1)
parser = argparse.ArgumentParser(description='Process parameters')
parser.add_argument('--epochs', default=50, type=int, dest='N_epochs', help='number of super-epochs where to save files')
parser.add_argument('--epoch_ini', default=1, type=int, dest='epoch_ini', help='number of initial epoch for saving purposes')
parser.add_argument('--filedata', default='', type=str, dest='filedata', help='name of the file with the data')
parser.add_argument('--filesave', default='', type=str, dest='filesave', help='name of the file where to save data')
parser.add_argument('--str_app', default='', type=str, dest='str_app', help='string to tag the training')
parser.add_argument('--validation_split', default=0.1, type=float, dest='validation_split', help='fraction of the data for validation')
parser.add_argument('--N_h', default=500, type=int, dest='N_h', help='number of neurons in the network')
parser.add_argument('--act_gain', default=1, type=float, dest='actions_gain', help='how to weight actions representation with respect to observations')
parser.add_argument('--sparsity', default=9e-9,type=float, dest='sparsity_coeff', help='sparsity')
args = parser.parse_args()

# load the dataset
foldername = os.path.dirname(os.path.realpath(__file__))+'/'
filedata =  args.filedata
filesave = args.filesave
data_mat = scipy.io.loadmat(foldername+filedata)
weights = data_mat.get('weights')
observations = data_mat.get('observations')
observations = observations.astype('float32')
actions = data_mat.get('actions')
actions_onehot = onehot(actions)


# parameters
N_h = args.N_h
N_epochs = args.N_epochs
epoch_ini = args.epoch_ini
str_app = args.str_app
sparsity_coeff = args.sparsity_coeff
N_tot = int(observations.shape[0])
actions_gain = args.actions_gain
validation_split = args.validation_split
N_x = int(observations.shape[1])
N_a = int(np.unique(actions).shape[0])
col_fact = 1
dist_fact = 1
N_batch = 5
N_traj = 500
N_samples = int(N_tot/N_batch)

# create the training set with batched data
# y are shifted over by one --> predictive case
x = np.concatenate((observations, actions_gain*actions_onehot), axis = 1)[:-1]
y = observations[1:]
#y = observations[:-1] # non-predictive case (auto-encode)
x_batched=batch_time(x,N_traj)
y_batched=batch_time(y, N_traj)


## CREATE THE MODEL
model = Sequential()
model.add(SimpleRNN(	units = N_h, 
			batch_input_shape = (N_batch, N_traj, N_x+N_a), 
                        # kernel = feed-forward network weights (input weights)
			kernel_initializer = initializers.random_normal(stddev=0.02), 

                        # recurrent weights
			recurrent_initializer=initializers.identity(), 

                        # True: do not reset/reinitialize weights after each round of training?
			stateful = True,

                        # ??
			return_sequences=True,	

                        # neuron activation function
			activation='sigmoid',


			activity_regularizer = l1(sparsity_coeff)))

# ordering is important; i.e. is there a sense of time. For images, this is not important.
# Builds a wrapper class to Dense layer that helps handle time
# wrapper: adds on to a class; gives it more properties. In this case, it is stores some information on time.
model.add(TimeDistributed(Dense(N_x, kernel_initializer = initializers.random_normal(stddev=0.02),activation = 'linear')))

# ## Reinitialize weights to loaded ones
#model.layers[1].layer.bias.set_value(weights['out_bias'][0,0].reshape(N_x)) 
#model.layers[1].layer.kernel.set_value(weights['out_weight'][0,0].T)
#model.layers[0].bias.set_value(weights['rec_bias'][0,0].reshape(N_h))
#model.layers[0].recurrent_kernel.set_value(weights['rec_weight'][0,0].T)
#model.layers[0].kernel.set_value(weights['inp_weight'][0,0].T)
model.layers[0].set_weights([ weights['inp_weight'][0,0].T, weights['rec_weight'][0,0].T,weights['rec_bias'][0,0].reshape(N_h)])
model.layers[1].set_weights([weights['out_weight'][0,0].T, weights['out_bias'][0,0].reshape(N_x)])


#rms = Adam(lr=0.001)
# rms = RMSprop(lr=0.0001, rho=0.95, epsilon=1e-07)#, decay=0.0,clipvalue = 0.5,clipnorm = 1)
# rms = RMSprop(lr=0.00001)#,clipvalue = 0.5,clipnorm = 1)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

## TRAIN THE MODEL AND SAVE THE DATA

# the callback call_stop is used to stop the training
# Would stop after N epochs if not given
call_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=15, verbose=0, mode='auto')
# the callback call_save saves the data in the end of each epoch
# i.e. the weights of network during learning; pass in a lambda function
call_save = LambdaCallback(on_epoch_end=lambda epoch, logs: 
        # save as MATLAB data file; layers[0] = recurrent; layers[1] = output
	scipy.io.savemat(filesave + 'weights_Ep'+ str(epoch+epoch_ini),	mdict={	  
                'out_bias':model.layers[1].get_weights()[1].reshape(N_x,1), 
                'out_weight':model.layers[1].get_weights()[0].T, 
		'rec_bias':model.layers[0].get_weights()[2].reshape(N_h,1), 
		'rec_weight':model.layers[0].get_weights()[1].T,
		'inp_weight':model.layers[0].get_weights()[0].T,
		'history': model.model.history.history}))

	        #  'out_weight':model.layers[1].layer.kernel.get_value().T, 
		# 'rec_bias':model.layers[0].bias.get_value().reshape(N_h,1), 
		# 'rec_weight':model.layers[0].recurrent_kernel.get_value().T,
		# 'inp_weight':model.layers[0].kernel.get_value().T,
		# 'history': model.model.history.history}))


# the callback call_lr reduces the learning ratio when learning doesn't improve
# Adjusting the learning rate after each epoch; decrease as learning goes on.

# pateince here must be > patience in call_stop
# adjust learning rate faster than stoping learning.
call_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.000001,verbose=1)

# train the model

# callback = function you call at beginning or end of each epoch; to act during learning and make adjustments to the
#           network
history = model.fit(x_batched, y_batched, batch_size = N_batch, epochs=N_epochs, validation_split = validation_split, verbose=2, shuffle = False, callbacks = [call_stop, call_save, call_lr])#
