import warnings
warnings.filterwarnings("ignore")

import sys
sys.executable
from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from filterpy.kalman import KalmanFilter

data = pd.read_csv("/home/dhruv/Inertial-Positioning-System/data.csv")
gyro_x = asanyarray(data.Gyro_X)
#scatter(data.Number,data.Gyro_X)

def kfilter(dt):
    kf = KalmanFilter(dim_x=2,dim_z=1)
    kf.F = array(([1,-dt],[0,1]))         #Transition Matrix
    kf.Q = eye(2)*0.001
    kf.B = 0
    kf.H = array(([1,0]))
    kf.R = 10
    kf.x = array(([0],[0]))
    kf.P = eye(2)*500
    return kf
    
tracker = kfilter(0.006)
zs=gyro_x
mu, cov, _, _ = tracker.batch_filter(zs)
plot(data.Number,mu[:,0])