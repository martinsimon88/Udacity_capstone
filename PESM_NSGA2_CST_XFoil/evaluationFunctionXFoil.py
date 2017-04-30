import XFoilGame
import numpy as np
import sympy as sp

sp.init_printing()


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
    testgen = XFG.newGame(actions, 'CST')

    return {
        "DD": testgen[0],
        "LL": testgen[1],
        "Dmin": testgen[2]
    }

def main(job_id, params):
    try:
        return TwoDXFoilGame(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in branin_con.py'
        return np.nan

'''
##test

#params = [0.5, 0.5, 0.5, 0.2, -0.1, 0.5, 0.3, 0.1]
params = [0.5, 0.5, 0.5, 0.2, -0.1, 0.5, 0.3, 0.1]

XFG = XFoilGame.XFoilGame()
testgen = XFG.newGame(params)
print testgen

'''