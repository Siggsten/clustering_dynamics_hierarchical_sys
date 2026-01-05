import math
import numpy as np
from functools import reduce
import time
from scipy.constants import Boltzmann as kB 
from tkinter import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from fractions import Fraction

#PARAMETERS
N = 200  # Number of particles.
L = 80  # Dimension of the squared arena.
v = 0.5  # Speed.
Rf = 2  # Flocking radius.
eta = 0.02  # Noise.
dt = 1  # Time step.
t_snapshots = np.array([1000*dt]) #times when to plot configurations
T_tot = t_snapshots[-1]#7500 #Total time
phi_list = [math.pi/2, math.pi/4, math.pi/6]
weight_list = [0,1]
rounds = 20

#FUNCTUIONS
def replicas(x, y, L):
    """
    Function to generate replicas of a single particle.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """    
    xr = np.zeros(9)
    yr = np.zeros(9)

    for i in range(3):
        for j in range(3):
            xr[3 * i + j] = x + (j - 1) * L #3i+j make a 2D representation of movements to 1D by (-L, +L),(0, +L),(+L, +L) etc
            yr[3 * i + j] = y + (i - 1) * L
    
    return xr, yr

def pbc(x, y, L):
    """
    Function to enforce periodic boundary conditions on the positions.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """   
    
    outside_left = np.where(x < - L / 2)[0] #np.where returns tuple -> take [0] to get index array for our 1D case
    x[outside_left] = x[outside_left] + L

    outside_right = np.where(x > L / 2)[0]
    x[outside_right] = x[outside_right] - L

    outside_up = np.where(y > L / 2)[0] #np.where returns tuple -> take [0] to get index array for our 1D case
    y[outside_up] = y[outside_up] - L

    outside_down = np.where(y < - L / 2)[0]
    y[outside_down] = y[outside_down] + L
    
    return x, y


def interaction(x, y, theta, Rf, L, hier_vec, wi, phi):
    """
    Function to calculate the orientation at the next time step.
    
    Parameters
    ==========
    x, y : Positions.
    theta : Orientations.
    Rf : Flocking radius.
    L : Dimension of the squared arena.
    s : Discrete steps.
    """
    
    N = np.size(x)
    theta_next = theta.copy()
    # Preselect what particles are closer than Rf to the boundaries.
    replicas_needed = reduce( 
        np.union1d, (
            np.where(y + Rf*np.sin(theta) > L / 2)[0], 
            np.where(y - Rf*np.sin(theta) < - L / 2)[0],
            np.where(x + Rf*np.cos(theta) > L / 2)[0],
            np.where(x - Rf*np.cos(theta) < - L / 2)[0]
        )
    )

    noise = np.zeros(N)
    for j in range(N):
        # Check if a particle j is too close to boundry so replicas are needed to calculate Rf correctly
        if np.size(np.where(replicas_needed == j)[0]):
            # Use replicas.
            xr, yr = replicas(x[j], y[j], L)
            nn = []
            for nr in range(9):
                dist2 = (x - xr[nr]) ** 2 + (y - yr[nr]) ** 2
                ang2 = np.arctan2((y - y[nr]),(x - x[nr]))
                ang2[j] = theta[j] #Set it to theta so it always end up in nn list
                nn = np.union1d(nn, np.where((dist2 <= Rf ** 2) & (ang2 >= theta-phi) & (ang2 <= theta+phi))[0])

        else:
            dist2 = (x - x[j]) ** 2 + (y - y[j]) ** 2
            ang2 = np.arctan2((y - y[j]),(x - x[j]))
            ang2[j] = theta[j] #Set it to theta so it always end up in nn list

            nn = np.where((dist2 <= (Rf) ** 2) & (ang2 >= theta-phi) & (ang2 <= theta+phi))[0]
        
        
        # The list of nearest neighbours is set.
        nn = nn.astype(int)
        
        #Implementing assignment specific behaviors
        if wi == 1: #Hierarchy
            j_vec = np.full(N, hier_vec[j]) #Matrix full of j's hierarchy
            weight_diff = hier_vec-j_vec #Weights as difference in hierarchy
            weight = np.array([int(x) if x > 0 else 0 for x in weight_diff]) #Set 0 to all elements <= 0
        
            if np.sum(weight[nn]) == 0:
                theta_next[j] = theta_next[j] + noise[j]

            else:
                av_sin_theta = np.average(np.sin(theta_next[nn]), weights = weight[nn])
                av_cos_theta = np.average(np.cos(theta_next[nn]), weights = weight[nn])
                theta_next[j] = np.arctan2(av_sin_theta, av_cos_theta) + noise[j]
        if wi == 0:
            weight = np.ones(N)
            av_sin_theta = np.average(np.sin(theta_next[nn]), weights = weight[nn])
            av_cos_theta = np.average(np.cos(theta_next[nn]), weights = weight[nn])
            theta_next[j] = np.arctan2(av_sin_theta, av_cos_theta) + noise[j]

    return theta_next


def global_alignment(theta):
    """
    Function to calculate the global alignment coefficient.
    
    Parameters
    ==========
    theta : Orientations.
    """
    
    N = np.size(theta)
    
    global_direction_x = np.sum(np.sin(theta))
    global_direction_y = np.sum(np.cos(theta))
        
    psi = np.sqrt(global_direction_x ** 2 + global_direction_y ** 2) / N
    
                   
    return psi



def area_polygon(vertices):
    """
    Function to calculate the area of a Voronoi region given its vertices.
    
    Parameters
    ==========
    vertices : Coordinates (array, 2 dimensional).
    """    
    
    N, dim = vertices.shape
    
    # dim is 2.
    # Vertices are listed consecutively.
    
    A = 0
    
    for i in range(N-1):
        # Below is the formula of the area of a triangle given the vertices.
        A += np.abs(
            vertices[- 1, 0] * (vertices[i, 1] - vertices[i + 1, 1]) +
            vertices[i, 0] * (vertices[i + 1, 1] - vertices[- 1, 1]) +
            vertices[i + 1, 0] * (vertices[- 1, 1] - vertices[i, 1])
        )
    
    A *= 0.5
    
    return A


def global_clustering(x, y, Rf, L):
    """
    Function to calculate the global alignment coefficient.
    
    Parameters
    ==========
    x, y : Positions.
    Rf : Flocking radius.
    L : Dimension of the squared arena.
    """
    
    N = np.size(x)
    
    # Use the replicas of all points to calculate Voronoi for 
    # a more precise estimate.
    points = np.zeros([9 * N, 2])

    for i in range(3):
        for j in range(3):
            s = 3 * i + j
            points[s * N:(s + 1) * N, 0] = x + (j - 1) * L
            points[s * N:(s + 1) * N, 1] = y + (i - 1) * L

    # The format of points is the one needed by Voronoi.
    # points[:, 0] contains the x coordinates
    # points[:, 1] contains the y coordinates
   
    vor = Voronoi(points)     
    '''
    vertices = vor.vertices  # Voronoi vertices.
    regions = vor.regions  # Region list. 
    # regions[i]: list of the vertices indices for region i.
    # If -1 is listed: the region is open (includes point at infinity).
    point_region = vor.point_region  # Region associated to input point.
    '''
   
    # Consider only regions of original set of points (no replicas).
    list_regions = vor.point_region[4 * N:5 * N]
    
    c = 0

    for i in list_regions:
        indices = vor.regions[i]
        # print(f'indices = {indices}')
        if len(indices) > 0:
            if np.size(np.where(np.array(indices) == -1)[0]) == 0:
                # Region is finite.
                # Calculate area.
                A = area_polygon(vor.vertices[indices,:])
                if A < np.pi * Rf ** 2:
                    c += 1                 
    c = c / N                  
    return c



def get_color(part_hier): #Function to map normalized value to color (blue -> red)
    "A function to create colorful particles"
    "The ones with high hiearchy are red"
    "The ones with low are blue"
    r = int(part_hier * 255)
    g = 0
    b = int((1 - part_hier) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'




#MAIN

N_skip = 1

c_values = np.zeros((len(weight_list), len(phi_list), T_tot+1, rounds))
psi_values = np.zeros((len(weight_list), len(phi_list), T_tot+1, rounds))
for w_index in range(len(weight_list)):
    wi = weight_list[w_index]
    for phi_index in range(len(phi_list)):
        phi = phi_list[phi_index]
        for round in range(rounds):

            #INITIALIZATION
            # Random position.
            x = (np.random.rand(N) - 0.5) * L  # in [-L/2, L/2]
            y = (np.random.rand(N) - 0.5) * L  # in [-L/2, L/2]
            # Random orientation.
            theta = 2 * (np.random.rand(N) - 0.5) * np.pi  # in [-pi, pi]
            # Vector of hierarchy
            #hier_vec = np.linspace(0, 1, N) #vector with N values from 0 to 1
            hier_vec = np.arange(1, N + 1, 1) #vector with N values from 1 to N
            norm_hier_vec = np.array([float((val - min(hier_vec)) / (max(hier_vec) - min(hier_vec))) for val in hier_vec])
            

            running = True  # Flag to control the loop.
            step = 0
            while running:
                
                # Calculate next theta from the rule.
                dtheta = eta * (np.random.rand(N) - 0.5) * dt

                ntheta = interaction(x, y, theta, Rf, L, hier_vec, wi, phi) + dtheta
                nx = x + v * np.cos(ntheta)
                ny = y + v * np.sin(ntheta)
                
                # Reflecting boundary conditions.
                nx, ny = pbc(nx, ny, L)

                if (step == t_snapshots).any():
                    for j in range(N):
                        plt.scatter(x[j], -y[j], color = get_color(norm_hier_vec[j]), label='Special Particles')
                    plt.quiver(x, -y, np.cos(ntheta), -np.sin(ntheta), width=0.001, headwidth=10)      
                    plt.xlim(-L/2, L/2)
                    plt.ylim(-L/2, L/2)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Configuration at time {step}')
                    plt.show()

                c_values[w_index, phi_index, step, round] = global_clustering(nx, ny, Rf, L)
                psi_values[w_index, phi_index, step,round] = global_alignment(theta)


                step += 1
                x[:] = nx[:]
                y[:] = ny[:]
                theta[:] = ntheta[:] 
                if step > T_tot:
                    running = False
        print(phi)
    print(wi)

plt.figure(figsize=(10, 5))
colors = ['orange', 'm', 'seagreen']
labels = ['ϕ = π/2', 'ϕ = π/4', 'ϕ = π/6']

c_mean = np.mean(c_values, axis=3)

# Hierarchy: w=0
# No hierarchy: w=1
for i in range(len(phi_list)):
    plt.plot(c_mean[0,i,:], '--', color=colors[i], linewidth=1, label=fr'{labels[i]}, No hierarchy')
    plt.plot(c_mean[1,i,:], color=colors[i], linewidth=1, label=fr'{labels[i]}, Hierarchy')

plt.title('Global clustering coefficient over time')
plt.legend()
plt.xlabel('time')
plt.ylabel('c')
plt.show()