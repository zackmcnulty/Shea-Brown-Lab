'''
This code follows from the example found at:
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

'''
import pandas
import time
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = pandas.read_csv('input_files/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
print(dataframe.head())
plt.plot(dataframe)
plt.show()

# ===============================

# fix the seed for reproducibility
numpy.random.seed(7)

dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize data
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
# i.e. for each time point, not only do I want to know the current value
# but values "lookback" steps in the past, giving the data this concept of time
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
# creates a series of X,Y pairs where X = value at time t, Y = value at time t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network

model = Sequential()

# generates a layer of cells that behave as our recurrent network part; they accept some input layer
# and perform a series of recurrent updates; Unlike other cell types, the LSTM cells store a "memory" of
# previous information flowing through them which affects their activity. Here we add 4 of these such
# nodes.
model.add(LSTM(4, input_shape=(1, look_back)))

# Dense(# of neurons in layer) is just adding a layer of your basic neurons; each neuron in this layer
# just takes in the outputs of the previous layer as inputs, weights these values according to some kernel (weight matrix)
# and applies an activation function
model.add(Dense(1))

# loss defines how you want to measure the error between predicted value and actual value in order to train the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions (un-normalize to get back in terms of thousands of passengers)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
