#Math Methods Recitation, November 9, 2017: Markov Chain Example

import numpy as np
import matplotlib.pyplot as plt

def MarkovMatrix(alpha, beta):
    #alpha = P(unemployed worker finds a job)
    #beta = P(employed worker loses a job)
    return np.array([[1-alpha, alpha],[beta, 1-beta]])

#Construct a particular matrix

alpha, beta = 0.3, 0.05
P = MarkovMatrix(alpha, beta)

#Simulate the Markov chain for T periods

states = np.array((0,1))
init_state = states[1] #Begin unemployed
num_periods = 250
chain = np.zeros(num_periods, dtype = int)
chain[0] = init_state
for i in range(1,num_periods):
    chain[i] = np.random.choice(states,1, p = P[chain[i-1]])
    
#Plot the resulting chain and the fraction of time employed   

fig, ax = plt.subplots(figsize = (8,6))
periods = np.arange(num_periods)
ax.plot(periods, chain, label = 'Employment State')
emp_frac = np.cumsum(chain)/(1+periods)
ax.plot(periods, emp_frac, label = 'Cumulative Employment')
ax.set_title('Markov Chain Simulation')
ax.set_xlabel('Time Period')
plt.legend(loc = 3)
plt.show()


#Plot the transition matrix for many periods ahead

np.set_printoptions(precision = 2)
for i in range(1,15):
    print('The %s period transition matrix is:\n %s'%(i, np.linalg.matrix_power(P,i)))
