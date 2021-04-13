import numpy as np
import numpy.linalg as la
import time
import json
import itertools
from tqdm import tqdm
from scipy.optimize import lsq_linear
from matplotlib import pyplot as plt

import sys
sys.path.append('..//')

from convexgeometry import walks

def zonoid_membership_def(x, A):
    #print(f"Testing A={A}, b={x}")
    #print(f"A.shape = {A.shape}, b.shape={x.shape}")
    x = x[0] if len(x.shape) == 2 else x
    res = lsq_linear(A, x, bounds=(-1,1))
    return res['cost'] < 1e-10

def cubePoints(dim):
    """Returns a 2^dim-by-dim matrix of vertices of the [-1,1] cube"""
    return np.array(list(set([j 
        for i in itertools.combinations_with_replacement([-1,1],dim)
        for j in itertools.permutations(i,dim)])))

N = 1000
step_size = 1
dim = 2

# The points of the zonotope
Averts = [[0,1],[1,0],[1,1]]
A = np.array(Averts).T
origin = np.array([0,0],dtype='float64')

# Do the Grid Walk
walker = walks.GridWalk(zonoid_membership_def, origin, space=step_size, A=A, delta=0.3)

points = np.zeros((N,dim))
p = np.zeros((1,dim)) # start stepping from the origin
for i in tqdm(range(0,N)):
    p = walker.step()
    #if( i%100 == 0 ):
    #    print(f"{i}..", end='')
    points[i,:] = p

if dim is 2:
    plt.figure(1)
    plt.title(f"Grid Walk Zonoid samples (step_size: {step_size})")
    plt.scatter(points[:,0], points[:,1])
    plt.savefig(f"img/grid_zonoid_{N}_{step_size}.png")
else: 
    print("Too high dimension to plot")


# Do hit and run
walker = walks.HitAndRunWalk(zonoid_membership_def, origin, space=step_size, A=A)

points = np.zeros((N,dim))
p = np.zeros((1,dim)) # start stepping from the origin
for i in tqdm(range(0,N)):
    p = walker.step()
    #if( i%100 == 0 ):
    #    print(f"{i}..", end='')
    points[i,:] = p

if dim is 2:
    plt.figure(0)
    plt.title(f"Hit-and-Run Zonoid samples (step_size: {step_size})")
    plt.scatter(points[:,0], points[:,1])
    plt.savefig(f"img/hitrun_zonoid_{N}_{step_size}.png")
else: 
    print("Too high dimension to plot")


# Do the Ball Walk
walker = walks.BallWalk(zonoid_membership_def, origin, space=step_size, A=A, delta=0.5)

points = np.zeros((N,dim))
p = np.zeros((1,dim)) # start stepping from the origin
for i in tqdm(range(0,N)):
    p = walker.step()
    #if( i%100 == 0 ):
    #    print(f"{i}..", end='')
    points[i,:] = p

if dim is 2:
    plt.figure(1)
    plt.title(f"Ball Walk Zonoid samples (step_size: {step_size})")
    plt.scatter(points[:,0], points[:,1])
    plt.savefig(f"img/ball_zonoid_{N}_{step_size}.png")
else: 
    print("Too high dimension to plot")


