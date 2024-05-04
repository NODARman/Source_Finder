import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x, y, v, t
from scipy.optimize import fsolve

# Generate random data
x = np.random.rand(10) * 100
y = np.random.rand(10) * 100
t = np.sort(np.random.rand(10) * 100)

# Scatter plot
for index, item in enumerate(x):
    plt.plot(x[index], y[index], '.', color='magenta')
    plt.text(x[index] + x[index] / 30, y[index],
             f'({str(round(x[index], 2))}; {str(round(y[index], 2))}; {str(round(t[index], 2))})', fontsize=8, color='blue')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Randomly Generated Points')
plt.grid(True)
plt.show()

def nonlinearEquation(w):
    F=np.zeros(3)
    F[0]=2*w[0]**2+w[1]**2+w[2]**2-15
    F[1]=w[0]+w[1]+2*w[2]-9
    F[2]=w[0]*w[1]*w[2]-6
    return F

# generate an initial guess
initialGuess=np.random.rand(3)

# solve the problem
solutionInfo=fsolve(nonlinearEquation,initialGuess,full_output=1)


xx=[2, 5, -7, 0]
yy=[1, -7, -3, 10]
tt=[4, 5, 9, np.sqrt(173)]

for index, item in enumerate(tt):
  tt[index]+=4

#x= [2 5 -7 0 2]; y= [ 1 -7 -3 10 17]; t= [4 5 9 sqrt(173) 20];


# System of equations
def f(vars):
  x,y,v,t = vars
  # loop in the future!

  e1 = (xx[0] - x)**2 + (yy[0] - y)**2 - v**2 * (tt[0] - t)**2
  e2 = (xx[1] - x)**2 + (yy[1] - y)**2 - v**2 * (tt[1] - t)**2
  e3 = (xx[2] - x)**2 + (yy[2] - y)**2 - v**2 * (tt[2] - t)**2
  e4 = (xx[3] - x)**2 + (yy[3] - y)**2 - v**2 * (tt[3] - t)**2

  return [e1,e2,e3,e4]

# Solve the system
x,y,v,t = fsolve(f, [1,1,1,1])
print(x,y,v,t)

r=10
k=f([round(x,r),round(y,r),round(v,r),round(t,r)])
print(k)
# Source: https://cmps-people.ok.ubc.ca/jbobowsk/Python/html/Jupyter%20Nonlinear.html
