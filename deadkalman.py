import warnings
warnings.filterwarnings("ignore")

import sys
sys.executable
from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from filterpy.kalman import KalmanFilter
from math import *

data = pd.read_csv("/home/dhruv/Inertial-Positioning-System/data_1.csv")
gyro_x = asanyarray(data.GyroX)
gyro_y = asanyarray(data.GyroY)
gyro_z = asanyarray(data.GyroZ)
acc = asanyarray(([data.AccX[0],data.AccY[0],data.AccZ[0]*10]))
acc = acc.reshape(3,1)

def kfilter(dt):
    kf = KalmanFilter(dim_x=2,dim_z=1)
    kf.F = array(([1,-dt],[0,1]))         #Transition Matrix
    kf.Q = array(([0.001,0],[0,0.01]))
    kf.B = 0
    kf.H = array([[1,0]])
    kf.R = 100
    kf.x = array(([0],[0]))
    kf.P = eye(2)*500
    return kf
    
tracker = kfilter(0.006)
mu_x, cov, _, _ = tracker.batch_filter(gyro_x)
mu_y, cov, _, _ = tracker.batch_filter(gyro_y)
mu_z, cov, _, _ = tracker.batch_filter(gyro_z)
pitch = mu_x[0,0]
roll = mu_y[0,0]
yaw = mu_z[0,0]

#Gravity Compensation
DCM = array(([cos(roll),sin(roll)*cos(pitch),sin(roll)*sin(pitch)],
            [-sin(roll),cos(roll)*cos(pitch),cos(roll)*sin(pitch)],
            [0,-sin(pitch),cos(pitch)]))
ginframe = dot(DCM,array(([0],[0],[9.8])))
lin_acc = acc - ginframe
lin_acc