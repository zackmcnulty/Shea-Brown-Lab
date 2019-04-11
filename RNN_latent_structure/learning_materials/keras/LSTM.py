'''
This code follows from the example found at:
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

'''
import pandas
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset = pandas.read_csv('input_files/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
print(dataset.head())
plt.plot(dataset)
plt.ion()
plt.show()
time.sleep(10)
