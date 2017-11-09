#Math Methods Recitation, November 9, 2017: Markov Chain Example

import numpy as np
import matplotlib.pyplot as plt

def MarkovMatrix(alpha, beta):
    
    '''
    Description: Constructs a stochastic matrix for employement transitions
    
    Inputs:
        alpha: Probability unemployed worker finds a job
        beta: Probability employed worker loses a job
    '''
    
    return np.array([[1-alpha, alpha],[beta, 1-beta]])

def Simulation(alpha, beta, x_0, T):
    
    '''
    Description: Plots the Markov Chain for T periods
    
    Inputs:
        alpha, beta: MC parameters (between 0 and 1)
        x_0: initial state (0 or 1)
        T: number of periods (positive integer)
    '''
    
    #Simulate the chain
    P = MarkovMatrix(alpha, beta)
    chain = np.zeros(T, dtype = int)
    chain[0] = x_0
    for i in range(1,T):
        chain[i] = np.random.choice(states,1, p = P[chain[i-1]])
        
    #Plot the chain and the fraction of time employed
    fig, ax = plt.subplots(figsize = (8,6))
    periods = np.arange(T)
    emp_frac = np.cumsum(chain)/(1+periods)
    ax.plot(periods, chain, label = 'Employment State')
    ax.plot(periods, emp_frac, label = 'Cumulative Employment')
    ax.set_title('Markov Chain Simulation')
    ax.set_xlabel('Time Period')
    plt.legend(loc = 3)
    plt.show()
    
def TransitionMatrix(alpha, beta, K):
    
    '''
    Description: Returns the K-period transition matrix
    
    Inputs:
        alpha, beta: MC parameters (between 0 and 1)
        K: number of periods (positive integer)
    '''
    
    P = MarkovMatrix(alpha, beta)
    return np.linalg.matrix_power(P,i)
    
#Run a simulation
Simulation(.5,.25,0,250)

#Plot the transition matrix for several periods ahead:
np.set_printoptions(precision = 2)
for i in range(1,10):
    print('The %s period transition matrix is:\n %s'%(i, TransitionMatrix(0.5,0.25,i)))
