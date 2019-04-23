'''
The task is to build an autoencoder for the MNIST database. This will learn
to form a compressed representation of the handwritten digit (encoding)
and then decode it back into the original image.

Here, I follow the tutorial found at:

    https://blog.keras.io/building-autoencoders-in-keras.html


'''
# Input layers are self-explanatory: they simply store the input data that
#       will be fed to later layers
# Dense layers are the typical neuron in a neural network. They have a weight connecting
#       them to each neuron in the previous layer and they choose their output/activation
#       based on the canonical --> output = activation_func(Weights * activations_prev_layer + bias)
from keras.layers import Input, Dense # types of neurons we will use

from keras.models import Model, Sequential

sequential = False # use the Sequential Model class or the Model class; both have same results


# size of our low-dimensional representation. The 784 pixels values of
# each MNIST image will be compressed down to this.
encoding_dim = 32


# this is our input placeholder: where data (MNIST images) will be fed into
input_img = Input(shape=(784,))


# The encoding layer of our neural network. The activations in this layer store a
# condensed representation of the image. For these neurons, the RELU activation
# function is used. Some other parameters to play with
#       - use_bias: use or don't use a bias term
#       - kernel/bias_initializer: how to set the initial weights/bias
#       - kernel/bias regularizer: how to regularize values in the weights/bias
#       - activity regularizer: limit the activity of neurons in a layer; i.e. try to promote sparsity
#                               to have as few active neurons as possible
#       - kernel/bias contraints: limits to place on weights/bias values

# the (input_img) at the end passes in "input_img" as a parameter to the function returned by Dense
# This is an example of higher-order functions in python
# i.e. here the input to the layer is the image
encoded = Dense(encoding_dim, activation='relu')(input_img)

# decoded is the reconstructed image of the handwritten digit. Due to compression, it is likely lossy
# i.e. here the input to the layer is the encoded (low dimensional) image representation
# since this will be the output layer, we want to convert it back to the expected output format:
# an image (of 784 pixels) with pixel values in [0,1] (hence the sigmoid activation)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(inputs = input_img, outputs = decoded)

# here we set features about how the model is trained/optimized to perform its task and how
# we calculate the error in our model (i.e. the cost/loss functions)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#============================================================
# Same model as above but using Sequential() model instead of general model class

encoding_dim = 32

model = Sequential()
model.add(Dense(encoding_dim, input_dim = 784, activation='relu'))
model.add(Dense(784, activation = 'sigmoid'))
model.compile(optimizer='adadelta', loss='binary_crossentropy')
#==============================================================


# Time to load the data!
from keras.datasets import mnist
import numpy as np

# normally this data comes with labels, i.e. it would be in the form of (x,y) pairs with x being the image
# and y being the (one hot) label, but here since we are just autoencoding we don't care about the labels
# the temp variable _ simply discards them
(x_train, _), (x_test, _) = mnist.load_data()

# normalize data to be in [0,1] and flatten the images into vectors
x_train = x_train.astype('float32')/ 255.0   # convert from uint8 to float32
x_test = x_test.astype('float32')/ 255.0   # convert from uint8 to float32
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

print(x_train.shape)
print(x_test.shape)


# Train the data! Specify training time, batch size, and other various features
if not sequential:
    autoencoder.fit(x_train, x_train, # our desired output matches the input
                    epochs = 50, 
                    batch_size = 256, 
                    shuffle = True,
                    validation_data = (x_test, x_test))
else:
    model.fit(x_train, x_train, # our desired output matches the input
              epochs = 50, 
              batch_size = 256, 
              shuffle = True,
              validation_data = (x_test, x_test))



# Once the model is trained, we can use it to predict some stuff! We really should not use
# the training data for this prediction as it won't give us a fair representation of the error.
# Instead, we will use the test data set as its not incorporated into our model (extrapolation watch out)

if not sequential:
    predicted_images = autoencoder.predict(x_test)
else:
    predicted_images = model.predict(x_test)




# plot stuff
import matplotlib.pyplot as plt


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


