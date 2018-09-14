import warnings
warnings.filterwarnings("ignore")
from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from math import *

data = pd.read_csv("/home/dhruv/Downloads/imu.csv")
gyro_x = asanyarray(data.Gyro_X)
gyro_y = asanyarray(data.Gyro_Y)
gyro_z = asanyarray(data.Gyro_Z)
acc_x = asanyarray(data.Acc_X)
acc_y = asanyarray(data.Acc_Y)
acc_z = asanyarray(data.Acc_Z)


def fx(x,dt):
    y = x
    return y

def hx(x):
    return x

points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=1)

def unscentedfilter(dt,fx,hx,points):
    kf = UnscentedKalmanFilter(dim_x = 2,dim_z=1,dt=dt,hx=hx,fx=fx,points=points)
    kf.Q = array(([100,0],[0,0.1]))
    kf.R = 70          #More R means smoother curve
    kf.P = eye(2)*500
    return kf

def trap(a,b,fx):
    n = len(fx)
    delx = 0.006
    xi = linspace(a,b,n)
    result = zeros((n,1))
    result[0] = 0
    for i in range(1,len(result)):
        result[i] = ((fx[i-1]+fx[i])*delx/2)+result[i-1]
    return result

ukf = unscentedfilter(0.006,fx,hx,points)
mu_gyrox,cov = ukf.batch_filter(gyro_x)
mu_gyrox,_,_ = ukf.rts_smoother(mu_gyrox,cov)
mu_gyroy,cov = ukf.batch_filter(gyro_y)
mu_gyroy,_,_ = ukf.rts_smoother(mu_gyroy,cov)
mu_gyroz,cov = ukf.batch_filter(gyro_z)
mu_gyroz,_,_ = ukf.rts_smoother(mu_gyroz,cov)

mu_accx,cov = ukf.batch_filter(acc_x)
mu_accx,_,_ = ukf.rts_smoother(mu_accx,cov)
mu_accy,cov = ukf.batch_filter(acc_y)
mu_accy,_,_ = ukf.rts_smoother(mu_accy,cov)
mu_accz,cov = ukf.batch_filter(acc_z)
mu_accz,_,_ = ukf.rts_smoother(mu_accz,cov)

accx_filtered = array(([mu_accx[:,0]]))
accy_filtered = array(([mu_accy[:,0]]))
accz_filtered = array(([mu_accz[:,0]]))
gyrox_filtered = mu_gyrox[:,0]
gyroy_filtered = mu_gyroy[:,0]
gyroz_filtered = mu_gyroz[:,0]

roll = trap(0,len(data.Number),gyrox_filtered)
pitch = trap(0,len(data.Number),gyroy_filtered)
yaw = trap(0,len(data.Number),gyroz_filtered)
lin_accx = zeros((len(accx_filtered.T),1))
lin_accy = zeros((len(accy_filtered.T),1))
lin_accz = zeros((len(accz_filtered.T),1))

for i in range(0,len(acc_x)):
    if pitch[i,0] < 0.015:
        pitch[i,0]=0
    if roll[i,0] < 0.015:
        roll[i,0]=0
    #Gravity Compensation
    ginframe = array(([sin(pitch[i,0])],
          [cos(pitch[i,0])*sin(roll[i,0])],
          [cos(roll[i,0])*cos(pitch[i,0])]))
    lin_accx[i,0] = accx_filtered[0,i] - ginframe[0]
    lin_accy[i,0] = accy_filtered[0,i] - ginframe[1]
    lin_accz[i,0] = -1*(accz_filtered[0,i] - ginframe[2])

Vel_x = trap(0,len(data.Number),lin_accx)
Vel_y = trap(0,len(data.Number),lin_accy)
Vel_z = trap(0,len(data.Number),lin_accz)
Pos_x = trap(0,len(data.Number),Vel_x)
Pos_y = trap(0,len(data.Number),Vel_y)
Pos_z = trap(0,len(data.Number),Vel_z)

subplot(231)
plot(data.Number,Pos_x)
title("Pos_X")
subplot(232)
plot(data.Number,Pos_y)
title("Pos_Y")
subplot(233)
plot(data.Number,Pos_z)
title("Pos_Z")
subplot(234)
plot(data.Number,roll*180/pi)
title("Roll")
subplot(235)
plot(data.Number,pitch*180/pi)
title("Pitch")
subplot(236)
plot(data.Number,yaw*180/pi)
title("Yaw")
show()
raw_input