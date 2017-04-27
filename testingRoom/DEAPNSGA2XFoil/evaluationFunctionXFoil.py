import time, array, random, copy, math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] ='\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

import seaborn
seaborn.set(style='whitegrid')
seaborn.set_context('notebook')

from deap import algorithms, base, benchmarks, tools, creator

#To make XFoil work:

import os
import string
import deap
from deap import tools, base
import random
import pandas
import numpy as np
import XFoilGame

#Go to directory
os.chdir("/home/simonx/Documents/Udacity/Projects/capstone/PESMOC_XFoil/Archive")


#For acquisition function
import numpy as np
import sympy as sp
sp.init_printing()
from numpy import sin, exp, cos
from sklearn.preprocessing import normalize
from scipy.stats import norm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


def TwoDXFoilGame(job_id, params):
    #print individual
    LEU = params['LEU']
    U1 = params['U1']
    U2 = params['U2']
    TEU = params['TEU']
    LED = params['LED']
    D1 = params['D1']
    D2 = params['D2']
    TED = params['TED']

    actions = [LEU, U1, U2, TEU, LED, D1, D2, TED]
    XFG = XFoilGame.XFoilGame()
    testgen = XFG.newGame(actions)

    return {
        "LL": testgen[0],
        "DD": testgen[1],
    }

def main(job_id, params):
    try:
        return TwoDXFoilGame(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in branin_con.py'
        return np.nan

#TwoDXFoilGame([0.06699999999999999, 0.028999999999999998])