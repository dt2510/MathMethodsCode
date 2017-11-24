'''
Title: Math Methods Recitation, Monday, November 20, 2017
Author: David Thompson
Description: Calculates a deterministic optimal growth (OG) model and plots value/policy function iteration
Note: Adapted from QuantEcon (https://lectures.quantecon.org/py/optgrowth.html)
'''

import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

#Create a class for the model------------------------------------------------------
class OptimalGrowth:
    '''
    Log-linear, deterministic optimal growth model
    '''
    
    def __init__(self, alpha = 0.4, beta = 0.96):
        '''
        Parameters:
            alpha = Cobb-Douglas parameter
            beta = Discount factor
        '''
        self.alpha, self.beta = alpha, beta
        
    def u(self, c):
        #Utility
        return np.log(c)

    def f(self, k):
        #Production function
        return k**self.alpha

#Construct the Bellman Operator ---------------------------------------------------
def bellman_operator(w, grid, beta, u, f):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function Tw and the policy function on the grid points.

    Parameters
    ----------
    w : array_like(float, ndim=1)
        The value of the input function on different grid points
    grid : array_like(float, ndim=1)
        The set of grid points
    beta : scalar
        The discount factor
    u : function
        The utility function
    f : function
        The production function
    """
    
    #Apply linear interpolation to w
    w_func = lambda x: np.interp(x, grid, w)

    #Initialize Tw and sigma
    Tw = np.empty_like(w)
    sigma = np.empty_like(w)

    #Set Tw[i] = max_c { u(c) + beta*w(f(y  - c))}
    for i, y in enumerate(grid):
        def objective(c):
            return - u(c) - beta * w_func(f(y - c))
        c_star = fminbound(objective, x1 = 1e-10, x2 = y, xtol = 1e-6)
        sigma[i] = c_star
        Tw[i] = - objective(c_star)

    return Tw, sigma
    
#Initialize a model from the class ---------------------------------------------------
Model = OptimalGrowth()

#Construct a grid of states
grid_max = 4         # Largest grid point
grid_size = 200      # Number of grid points
grid = np.linspace(1e-5, grid_max, grid_size)

#Initialize Plot Area ----------------------------------------------------------------
fig, ax = plt.subplots(1,2,figsize=(9, 6))
ax[0].set_ylim(-40, 10)
ax[1].set_ylim(0, 4)
ax[0].set_xlim(np.min(grid), np.max(grid))
ax[1].set_xlim(np.min(grid), np.max(grid))
ax[0].set_title('Convergence of Value Function')
ax[1].set_title('Converge of Optimal Policy Correspondence')

#Iterate value function from an initial guess and plot -------------------------------
w = grid  # An initial guess
n = 25 #Iterate 25 times
lb = 'initial condition'
ax[0].plot(grid, w, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)

for i in range(n):
    w, pol = bellman_operator(w,
                         grid,
                         Model.beta,
                         Model.u,
                         Model.f
                        )
    ax[0].plot(grid, w, color=plt.cm.jet(i / n), lw=2, alpha=0.6)
    ax[1].plot(grid, pol, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax[0].legend(loc='lower right')
plt.show()

#Plot one particular path given the policy function above ---------------------------

pol_func = lambda x: np.interp(x, grid, pol)

time_periods = 100
t = np.arange(time_periods)
consumption = np.zeros(time_periods)
y = np.zeros(time_periods)
y[0] = 1
for i in range(time_periods-1):
    consumption[i] = pol_func(y[i])
    y[i+1] = Model.f(y[i] - consumption[i])
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(t[0:time_periods-1], y[0:time_periods-1], label='y')
ax.plot(t[0:time_periods-1], consumption[0:time_periods-1], label='Consumption')
ax.set_title('State variable and consumption over time')

plt.legend()
plt.show()
