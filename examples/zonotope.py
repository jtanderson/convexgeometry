import numpy as np
import numpy.linalg as la
import time
import json
import itertools
from tqdm import tqdm
from scipy.optimize import lsq_linear
from matplotlib import pyplot as plt
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

walker = walks.HitAndRunWalk(zonoid_membership_def, np.array([0,0]), space=step_size, A=A)

points = np.zeros((N,dim))
p = np.zeros((1,dim)) # start stepping from the origin
for i in tqdm(range(0,N)):
    p = walker.step()
    #if( i%100 == 0 ):
    #    print(f"{i}..", end='')
    points[i,:] = p

if dim is 2:
    plt.title(f"Zonoid samples (step_size: {step_size})")
    plt.scatter(points[:,0], points[:,1])
    plt.savefig(f"img/zonoid_{N}_{step_size}.png")
else: 
    print("Too high dimension to plot")


