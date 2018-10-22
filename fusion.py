from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dt = 0.001 #Sample Rate for Accelerometer and Gyroscope is 1 KHz

data = pd.read_csv("/home/pavan/Inertial-Positioning-System/imu.csv")

GyroX = np.asanyarray(data.Gyro_X)
GyroY = np.asanyarray(data.Gyro_Y)
GyroZ = np.asanyarray(data.Gyro_Z)
AccX = np.asanyarray(data.Acc_X)
AccY = np.asanyarray(data.Acc_Y)
AccZ = np.asanyarray(data.Acc_Z)

l = len(GyroX)

def fx(x,dt):  #State Transformation Function
    return x

def hx(x):  #Measurement Function
    return x

def Unscentedfilter(zs):   # Filter function
    points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, fx=fx, hx=hx, points=points, dt=dt)
    ukf.Q = np.array(([1,0],[0,0.1]))
    ukf.R = 50
    ukf.P = np.eye(2)*500
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
            thetaf[i] = 0.0005*(Gyro[0] + Gyro[i])
        else:
            x = Gyro[i-1] + x
            thetaf[i] = 0.0005*(Gyro[0]+Gyro[i]+2*x)

    return thetaf

def filter(w,angle):
    A = np.array([[1, -dt],
                  [0, 1]], dtype = float )

    B = np.array([[dt],
                  [0]], dtype = float)
 
    A_transpose = A.transpose()

    H = np.array([[1, 0]], dtype = float)

    Q = np.array([[1, 0],
                  [0, 1]], dtype = float )

    Q = Q*dt

    R = np.array([[1]], dtype = float)

    P_previous_prior = np.array([[2.1, 0],
                                 [0, 2.1]], dtype = float)

    output = [0]*len(w)

    x = np.array([[0],
                  [0]], dtype = float)

    for i in range(0,l):
        z_present = angle[i]

        #Prediction
        x_current_priori = np.dot(A,x) + np.dot(B,w[i])
        a = np.dot(P_previous_prior,A_transpose)
        P_current_priori = np.dot(A, a) + Q

        #Update
        m = np.dot( np.dot(H , P_current_priori), H.transpose() )
        n = m + R
        K_current = np.dot( P_current_priori, H.transpose() ) / n
        x_current_posterior = x_current_priori + np.dot( K_current, z_present - np.dot(H,x_current_priori) )  
        P_current_posterior = P_current_priori - np.dot( np.dot( K_current, H ), P_current_priori)
        
        #Assignment of new variables
        output[i] = x_current_posterior[0]
        P_previous_prior = P_current_posterior
        x = x_current_posterior

    return output

def GravityComp(Acc_X,Acc_Y,Acc_Z,roll,pitch): #Gravity Compensator
    lin_AccX = [0] * l
    lin_AccY = [0] * l
    lin_AccZ = [0] * l
    g = np.array([0, 0, 1], dtype = float)
    for i in range(0,l):
        DCM = np.array([ [np.cos(pitch[i]), np.sin(roll[i])*np.sin(pitch[i]), -np.cos(roll[i])*np.sin(pitch[i])],
                         [0 , np.cos(roll[i]), np.sin(roll[i])],
                         [np.sin(pitch[i]), -np.sin(roll[i])*np.cos(pitch[i]), np.cos(roll[i])*np.cos(pitch[i])] ], dtype = float)
        Acc = np.array([[Acc_X[i], Acc_Y[i], Acc_Z[i]]], dtype = float)
        Acc_Inertial = np.dot(Acc,DCM)
        final_acc = Acc_Inertial - g
        lin_AccX[i] = final_acc[0,0]
        lin_AccY[i] = final_acc[0,1]
        lin_AccZ[i] = final_acc[0,2]
    return lin_AccX,lin_AccY,lin_AccZ 

def Velocity(Accl):  #Velocity Calculator
    Vf = [0] * l
    x = 0

    for i in range(0,l):    
        if i == 0:
            Vf[i] = 0
        elif i == 1:
            Vf[i] = 0.0005*( 9.81*(Accl[0] + Accl[i]) )
        else:
            x = 9.81*Accl[i-1] + x
            Vf[i] = 0.0005*( 9.81*(Accl[0] + Accl[i]) + (2*x) )

    return Vf

def Position(Vel):  #Position Calculator
    X = [0] * l
    x = 0

    for i in range(0,l):    
        if i == 0:
            X[i] = 0
        elif i == 1:
            X[i] = 0.0005*(Vel[0] + Vel[i])
        else:
            x = Vel[i-1] + x
            X[i] = 0.0005*(Vel[0]+Vel[i]+2*x)

    return X

def Measurement_Pitch(a,b,c): #Calculating Angles from accelerometer
    d = [0]*len(a)
    for i in range(0,len(a)):
        d[i] = np.arctan(-a[i]/np.sqrt(c[i]*c[i] + b[i]*b[i]))
    return d

def Measurement_Roll(a,b,c): #Calculating Angles from accelerometer
    d = [0]*len(a)
    for i in range(0,len(a)):
        d[i] = np.arctan(a[i]/np.sqrt(c[i]*c[i] + b[i]*b[i]))
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
GravityCompAccX, GravityCompAccY, GravityCompAccZ = GravityComp(FilteredAccX,FilteredAccY,FilteredAccZ,AngleX,AngleY)

print("Calculating Velocity")
VelX = Velocity(GravityCompAccX)
VelY = Velocity(GravityCompAccY)
VelZ = Velocity(GravityCompAccZ)

print("Calculating Velocity")
PosX = Position(VelX)
PosZ = Position(VelZ)
PosY = Position(VelY)

print("Calculating Postion")
plt.plot(data.index,GyroY,'r')
plt.plot(data.index,AngleY,'g')
plt.plot(data.index,pitch,'b')
plt.plot(data.index,GravityCompAccX,'y')
plt.plot(data.index,PosX,'g')
plt.grid()
plt.show()
