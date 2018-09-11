from scipy import integrate
from numpy import *
from matplotlib.pyplot import *
from pandas import *

data = read_csv("data1.csv")

def trap(a,b,fx):
    n = len(fx)
    delx = (b-a)/n
    xi = linspace(a,b,n)
    result = zeros((n,1))
    result[0] = -1
    for i in range(1,len(result)):
        result[i] = ((fx[i-1]+fx[i])*delx/2)+result[i-1]
    return result

x = linspace(0,3.14,100)
x = x.reshape((100,1))
a = sin(x)
apr = -1*cos(x)
b = trap(0,3.14,a)
plot(x,b,'b')
plot(x,a,'r')
plot(x,apr,'g')
show()
raw_input