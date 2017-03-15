from Gen2Airfoil import *
from testXFoil import *
import matplotlib.pyplot as plt
import random
import numpy as np
import os

# remove all files from last run
filelist = [ f for f in os.listdir(".") if f.endswith(".dat") ]
for f in filelist:
    os.remove(f)
#    print f
filelist = [ f for f in os.listdir(".") if f.endswith(".log") ]
for f in filelist:
    os.remove(f)

# Airfoil is defined by
#  LEU = Leading edge up            LED = Leading edge down
#  C25 = Camber at 25%              T25 = Camber at 25%
#  C50 = Camber at 50%              T50 = Camber at 50%
#  C75 = Camber at 75%              T75 = Camber at 75%

#              LEU   LED     C25   C50    C75      T25   T50   T75
genmaxs = [    0.2,  0.2,    0.2,  0.2,   0.2,     0.2,  0.2,  0.1   ]
genmins = [    0.01,  0.01,    0.01,  0.01,   0.01,     0.01,  0.01,  0.01   ]

ngen= len (genmaxs)

foilnum = 0

class XFoilGame():

    def __init__(self):
        self.child = [0,0,0,0,0,0,0,0]
        self.step=0.05
        self.Re = 500
        self.M = 0.1
        self.NoIter = 200
        self.Ncrit = 9.0
        self.state = None

    def newborn(self, gen):
        global foilnum
        foilnum+=1
        return [0,gen,'%06d' %foilnum]

    def check_range(self, gen):
        for i in range(0,ngen):
            if(gen[i]>genmaxs[i]):
                gen[i]=genmaxs[i]
            if(gen[i]<genmins[i]):
                gen[i]=genmins[i]



    def takeAction(self, actions):

        if actions % 2 == 0:
            step = -self.step
        else:
            step = self.step
        position = int(np.ceil(actions/2))
        self.child[position]=self.child[position]+step

        self.check_range(self.child)
        print self.child
        return self.newborn(self.child)



    def newGame(self, actions):
        testgen = self.takeAction(actions)
        name=testgen[2]
        gen2airfoil(testgen[1],name)
        Xfoil(testgen[2],self.Ncrit,self.Re, self.M, self.NoIter)
        print name,getLDmax(name)

    def reset(self):
        self.a

        return state

for i in range(1, 10, 1):
    actions = np.random.random_integers(0,15,1)
    XFG=XFoilGame()
    testgen = XFG.newGame(actions)


print testgen

