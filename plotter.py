import pandas as pd
from matplotlib.pyplot import *
from numpy import *
from pykalman import KalmanFilter


data = pd.read_csv('imu.csv')

F = array([[1.0,-0.00001],[0,1]])
H = array([1,0])
kf = KalmanFilter(transition_matrices = F,observation_matrices = H)
measurements = asarray([data.Gyro_X])
kf = kf.em(measurements,n_iter = 3)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
plot(data.Number,data.Gyro_X)
plot(data.Number,smoothed_state_means[:,1],'r')
show()
#f = figure(1)
#scatter(data.Number,data.Gyro_X)
#f.show()
#Frequncy Domain
#gyrox = data.Gyro_X
#fftdata = fft.fft(gyrox.real)
#freq = fft.fftfreq(gyrox.shape[-1])
#g = figure(2)
#scatter(freq,fftdata)
#g.show()
raw_input()