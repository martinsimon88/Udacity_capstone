'''ParetoOptimizer contains the main system of Pareto front based decision making for Multi-objective aerodynamic optimization'''

'''
1. Initial design
2. Objective function evaluations
3. Identify non-dominated points
4. Fit RBF surrogate models to the data
5. Select sample point on the basis of the two former steps
    - Decision beteen global and local search
    - Back to step 2'''

import os
import string
import deap
from deap import tools, base
import random
import pandas
import numpy as np
import XFoilGame

os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/Pareto/Archive")


'''
#Initiate XFoilGame
for i in range(1, 3, 1):
    actions = np.random.rand(8)
    #actions=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    XFG=XFoilGame.XFoilGame()
    testgen = XFG.newGame(actions)
'''

#Import initial dataset
txtallResults = open("allResults.log", 'r')
txtallResults = txtallResults.readlines()

txtallActions = open("allActions.log", 'r')
txtallActions = txtallActions.readlines()

alpha=[]
CL=[]
CD=[]
CDp=[]
CM=[]
temp=[]
actions = []
for i in range(0, len(txtallResults)):
    # print flines[i]
    words = string.split(txtallResults[i])
    alpha.append(float(words[1]))
    CL.append(float(words[2]))
    CD.append(float(words[3]))
    CDp.append(float(words[4]))
    CM.append(float(words[5]))

print max(CL)

for i in range(0, len(txtallActions)):
    words2 = string.split(txtallActions[i])
    for j in range(0, 8):
        temp.append(float(words2[j+1]))
    actions.append(temp)

#If necessary do objective function evaluations for the initial dataset



#Identify non-dominated points

def pareto_dominance(x, y):
    return tools.emo.isDominated(x.fitness.values, y.fitness.values)


#Use RBF to find estimate the new intials Gaussian?



#Make decision of which points to evaluate with acquisition function?


#Where does the GA come in to play?