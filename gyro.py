import smbus
import time
import pandas as pd
import csv

#Registars of MPU6050
pwr_mgmt = 0x6B
smplrt = 0x19
config = 0x1A
gyro_config = 0x1B
int_enable = 0x38
acc_xout = 0x3B
acc_yout = 0x3D
acc_zout = 0x3F
gyr_xout = 0x43
gyr_yout = 0x45
gyr_zout = 0x47
offset_Ax = 0
offset_Ay = 0
offset_Az = 0

offset_gyrox = 0
offset_gyroy = 0
offset_gyroz = 0

with open('imu.csv','wb') as csvfile:
	file = csv.writer(csvfile, delimiter=',', dialect='excel')
	file.writerow(['Number','Gyro_X','Gyro_Y','Gyro_Z','Acc_X','Acc_Y','Acc_Z'])

def MPU_init():
	bus.write_byte_data(dev_add,smplrt,7)
	bus.write_byte_data(dev_add,pwr_mgmt,1)
	bus.write_byte_data(dev_add,config,1)
	bus.write_byte_data(dev_add,gyro_config,24)
	bus.write_byte_data(dev_add,int_enable,1)

def raw_data(addr):
	high = bus.read_byte_data(dev_add,addr)
	low  = bus.read_byte_data(dev_add,addr+1)

	value = ((high << 8) | low)

	if (value > 32768):
		value = value - 65536
	return value

bus = smbus.SMBus(1)
dev_add = 0x68	#Device Address

MPU_init()

print("Calibrating Device")

for i in range(1,2001):
	#Accelerometer
        acc_x = raw_data(acc_xout)
        acc_y = raw_data(acc_yout)
        acc_z = raw_data(acc_zout)

        #Gyroscope
        gyro_x = raw_data(gyr_xout)
        gyro_y = raw_data(gyr_yout)
        gyro_z = raw_data(gyr_zout)

        Ax = acc_x/16384.0/2000.0
        Ay = acc_y/16384.0/2000.0
        Az = acc_z/16384.0/2000.0

        p = gyro_x/131.0/2000.0
        q = gyro_y/131.0/2000.0
        r = gyro_z/131.0/2000.0

	offset_Ax = offset_Ax + Ax
	offset_Ay = offset_Ay + Ay
	offset_Az = offset_Az + Az

	offset_gyrox = offset_gyrox + p
	offset_gyroy = offset_gyroy + q
	offset_gyroz = offset_gyroz + r

print("Calibirated")

print("Reading Data")

i = 0

while True:

	i = i + 1
	#Accelerometer
	acc_x = raw_data(acc_xout)
	acc_y = raw_data(acc_yout)
	acc_z = raw_data(acc_zout)

	#Gyroscope
	gyro_x = raw_data(gyr_xout)
	gyro_y = raw_data(gyr_yout)
	gyro_z = raw_data(gyr_zout)

	Ax = acc_x/16384.0 - offset_Ax
	Ay = acc_y/16384.0 - offset_Ay
	Az = acc_z/16384.0 - offset_Az + 1

	p = gyro_x/131.0 - offset_gyrox
	q = gyro_y/131.0 - offset_gyroy
	r = gyro_z/131.0 - offset_gyroz

	list = [i,p,q,r,Ax,Ay,Az]
	with open("imu.csv","a") as data:
		wr = csv.writer(data, dialect='excel')
		wr.writerow(list)

	print("Gx = %.2f" %p,"\t Gy = %.2f" %q, "\t Gz=%.2f" %r, "\t Ax=%.2f g" %Ax, "\t Ay=%.2f g" %Ay, "\t Az=%.2f g" %Az)
