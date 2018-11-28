from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *

dt = 0.001 #Sample Rate for Accelerometer and Gyroscope is 1 KHz

data = pd.read_csv("/home/pavan/Desktop/Inertial-Positioning-System/imu.csv")

GyroX = asanyarray(data.Gyro_X)
GyroY = asanyarray(data.Gyro_Y)
GyroZ = asanyarray(data.Gyro_Z)
AccX = asanyarray(data.Acc_X)
AccY = asanyarray(data.Acc_Y)
AccZ = asanyarray(data.Acc_Z)

l = len(GyroX)

"""for i in range(0,l):
    if -0.01 < AccX[i] < 0.01:
            AccX[i] = 0
    if -0.01 < AccY[i] < 0.01:
            AccY[i] = 0
    if -0.01 < AccZ[i] < 0.01:
            AccZ[i] = 0"""

def fx(x,dt):  #State Transformation Function
    return x

def hx(x):  #Measurement Function
    return x

def Unscentedfilter(zs):   # Filter function
    points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, fx=fx, hx=hx, points=points, dt=dt)
    ukf.Q = array(([50, 0],
                   [0, 50]))
    ukf.R = 100
    ukf.P = eye(2)*2
    mu, cov = ukf.batch_filter(zs)

    x,_,_ = ukf.rts_smoother(mu, cov)

    return x[:,0]

def Angle(Gyro):  # Angle Calculator
    thetaf = [0] * l
    x = 0

    for i in range(0,l):    
        if i == 0:
            thetaf[i] = 0
        elif i == 1:
            thetaf[i] = (dt/2)*(Gyro[0] + Gyro[i])
        else:
            x = Gyro[i-1] + x
            thetaf[i] = (dt/2)*(Gyro[0]+Gyro[i]+2*x)

    return thetaf

def filter(w,angle):
    A = array([[1, dt],
               [0, 1]], dtype = float )

    B = array([[0],
               [1]], dtype = float)

    A_transpose = A.transpose()

    H = array([[1, 0]], dtype = float)

    Q = array([[50, 0],
               [0, 50]], dtype = float )

    R = array([[0.01]], dtype = float)

    P_previous_prior = array([[2, 0],
                              [0, 2]], dtype = float)

    output = [0]*l

    x = array([[0],
               [0]], dtype = float)
    
    for i in range(0,l):
        z_present = angle[i]

        #Prediction
        x_current_priori = dot(A,x) + dot(B,w[i])
        a = dot(P_previous_prior,A_transpose)
        P_current_priori = dot(A, a) + Q

        #Update
        m = dot( dot(H , P_current_priori), H.transpose() )
        n = m + R
        K_current = dot( P_current_priori, H.transpose() ) / n
        x_current_posterior = x_current_priori + dot( K_current, z_present - dot(H,x_current_priori) )  
        P_current_posterior = P_current_priori - dot( dot( K_current, H ), P_current_priori)
        
        #Assignment of new variables
        output[i] = x_current_posterior[0,0]
        P_previous_prior = P_current_posterior
        x = x_current_posterior

    return output

def GravityComp(Acc_X,Acc_Y,Acc_Z,roll,pitch): #Gravity Compensator
    lin_AccX = [0] * l
    lin_AccY = [0] * l
    lin_AccZ = [0] * l
    g = array([0, 0, -1], dtype = float)
    for i in range(0,l): 
        """if -0.01 < Acc_X[i] < 0.01:
            Acc_X[i] = 0
        if -0.01 < Acc_Y[i] < 0.01:
            Acc_Y[i] = 0
        if -0.01 < Acc_Z[i] < 0.01:
            Acc_Z[i] = 0""" 
        ginframe = array([ [sin(pitch[i])],
                           [cos(pitch[i])*sin(roll[i])],
                           [cos(roll[i])*cos(pitch[i])] ], dtype = float )
        #Acc = array([[Acc_X[i], Acc_Y[i], Acc_Z[i]]], dtype = float)
        #final_acc = Acc + dot(g,DCM)
        lin_AccX[i] = Acc_X[i] + ginframe[0,0]
        lin_AccY[i] = Acc_Y[i] + ginframe[1,0]
        lin_AccZ[i] = -Acc_Z[i] + ginframe[2,0]
    return lin_AccX,lin_AccY,lin_AccZ 

def Velocity(Accl):  #Velocity Calculator
    Vf = [0] * l
    x = 0

    for i in range(0,l):    
        if -0.02 < Accl[i] < 0.02:
            Accl[i] = 0
        if i == 0:
            Vf[i] = 0
        elif i == 1:
            Vf[i] = (dt/2)*( 9.81*(Accl[0] + Accl[i]) )
        else:
            x = 9.81*Accl[i-1] + x
            Vf[i] = (dt/2)*( 9.81*(Accl[0] + Accl[i]) + (2*x) )


    return Vf

def Position(Vel):  #Position Calculator
    Pf = [0] * l
    y = 0

    for i in range(0,l):  
        """if -0.05 < Vel[i] < 0.0:
            Vel[i] = 0"""
        if i == 0:
            Pf[i] = 0
        elif i == 1:
            Pf[i] = (dt/2)*(Vel[0] + Vel[i])
        else:
            y = Vel[i-1] + y
            Pf[i] = (dt/2)*(Vel[0] + Vel[i] + 2*y)
        

    return Pf

def PositionZ(Vel):  #Position Calculator
    Pf = [0] * l
    y = 0

    for i in range(0,l):  
        """if -0.05 < Vel[i] < 0.0:
            Vel[i] = 0"""
        if i == 0:
            Pf[i] = 0
        elif i == 1:
            Pf[i] = (dt/2)*(Vel[0] + Vel[i])
        else:
            y = Vel[i-1] + y
            Pf[i] = (dt/2)*(Vel[0] + Vel[i] + 2*y)

    return Pf

def Measurement_Pitch(a,b,c): #Calculating Angles from accelerometer
    d = [0]*len(a)
    for i in range(0,len(a)):
        d[i] = arctan(a[i]/sqrt(c[i]*c[i] + a[i]*a[i]))
    return d

def Measurement_Roll(a,b,c): #Calculating Angles from accelerometer
    d = [0]*len(a)
    for i in range(0,len(a)):
        d[i] = arctan(a[i]/sqrt(c[i]*c[i] + a[i]*a[i]))
    return d

print("Filtering Acclerometer")

FilteredAccX = Unscentedfilter(AccX)
FilteredAccY = Unscentedfilter(AccY)
FilteredAccZ = Unscentedfilter(AccZ)

print("Filtering Gyroscope")
pitch = Measurement_Pitch(FilteredAccX, FilteredAccY, FilteredAccZ)
roll = Measurement_Roll(FilteredAccY, FilteredAccX, FilteredAccZ)

AngleY = filter(GyroY,pitch)
AngleX = filter(GyroX,roll)
FilteredGyroZ = Unscentedfilter(GyroZ)

print("Calculating Angles")
#AngleX = Angle(FilteredGyroX)
#AngleZ = Angle(FilteredGyroZ)
AngleZ = Angle(FilteredGyroZ)

print("Compensating Gravity")
GravityCompAccX, GravityCompAccY, GravityCompAccZ = GravityComp(AccX,FilteredAccY,FilteredAccZ,AngleX,AngleY)

print("Calculating Velocity")
VelX = Velocity(GravityCompAccX)
VelY = Velocity(GravityCompAccY)
VelZ = Velocity(GravityCompAccZ)

print("Calculating Position")
PosX = Position(VelX)
PosY = Position(VelY)
PosZ = PositionZ(VelZ)

print(l)
f  = plt.figure(1)
plt.subplot(311)
plt.plot(data.index,AccX,'y', label="Raw Acc")
plt.plot(data.index,FilteredAccX,'r', label = "Filt. Acc")
plt.plot(data.index,VelX,'g', label = "Lin Vel")
plt.plot(data.index,GravityCompAccX,'b', label = "Lin. Acc")
plt.plot(data.index,PosX,'m', label = "Pos")
plt.title("X")
plt.legend()
plt.grid()
plt.subplot(312)
plt.plot(data.index,AccY,'y', label="Raw Acc")
plt.plot(data.index,FilteredAccY,'r', label = "Filt. Acc")
plt.plot(data.index,VelY,'g', label = "Lin Vel")
plt.plot(data.index,GravityCompAccY,'b', label = "Lin Acc")
plt.plot(data.index,PosY,'m', label = "Pos")
plt.title("Y")
plt.legend()
plt.grid()
plt.subplot(313)
plt.plot(data.index,AccZ,'y', label="Raw Acc")
plt.plot(data.index,FilteredAccZ,'r', label = "Filt. Acc")
plt.plot(data.index,VelZ,'g', label = "Lin Vel")
plt.plot(data.index,GravityCompAccZ,'b', label = "Lin Acc")
plt.plot(data.index,PosZ,'m', label = "Pos")
plt.title("Z")
plt.legend()
plt.grid()
f.show()

g  = plt.figure(2)
plt.subplot(311)
plt.plot(data.index,GyroX,'y', label="Raw GyroZ")
#plt.plot(data.index,FilteredGyroX,'r', label = "Filt. GyroZ")
plt.plot(data.index,AngleX,'g', label = "Angle X")
plt.title("X")
plt.legend()
plt.grid()
plt.subplot(312)
plt.plot(data.index,GyroY,'y', label="Raw GyroZ")
#plt.plot(data.index,FilteredGyroY,'r', label = "Filt. GyroZ")
plt.plot(data.index,AngleY,'g', label = "Angle Y")
plt.title("Y")
plt.legend()
plt.grid()
plt.subplot(313)
plt.plot(data.index,GyroZ,'y', label="Raw GyroZ")
plt.plot(data.index,FilteredGyroZ,'r', label = "Filt. GyroZ")
plt.plot(data.index,AngleZ,'g', label = "Angle Z")
plt.title("Z")
plt.legend()
plt.grid()
g.show()
#plt.plot(data.index,GyroX,'y', label="Raw Acc")

plt.show()