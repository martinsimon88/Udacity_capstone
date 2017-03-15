from Gen2Airfoil import *
from testXFoil import *
import matplotlib.pyplot as plt
import random
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
        self.a=0
        self.b=0
        self.c=0
        self.d=0
        self.e=0
        self.f=0
        self.g=0
        self.h=0
        self.step=0.05


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



    def breed_random(self, actions):
        if actions==0:
            self.a=self.a+self.step
        if actions==1:
            self.a=self.a-self.step
        if actions==2:
            self.b=self.b+self.step
        if actions==3:
            self.b=self.b-self.step
        if actions==4:
            self.c=self.c+self.step
        if actions==5:
            self.c=self.c-self.step
        if actions==6:
            self.d=self.d+self.step
        if actions==7:
            self.d=self.d-self.step
        if actions==8:
            self.e=self.e+self.step
        if actions==9:
            self.e=self.e-self.step
        if actions==10:
            self.f=self.f+self.step
        if actions==11:
            self.f=self.f-self.step
        if actions==12:
            self.g=self.g+self.step
        if actions==13:
            self.g=self.g-self.step
        if actions==14:
            self.h=self.h+self.step
        if actions==15:
            self.h=self.h-self.step

        child = [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h]
        self.check_range(child)
        print child
        return self.newborn(child)


Re = 500
M = 0.1
NoIter = 200
Ncrit = 9.0
actions = 5
XFG=XFoilGame()
testgen = XFG.breed_random(actions)
print testgen



#testgen = [ 0.05,  0.025,    0.1,  0.1,   0.05,     0.1,  0.05,  0.025   ]
#testgen = [ 0.0376,  0.0121,  0.0845, 0.1290,  0.0496,  0.0295,  0.1326,  0.0277   ]
#name = "myfoil_1111"
#gen2airfoil(testgen,name)

name=testgen[2]
gen2airfoil(testgen[1],name)
Xfoil(testgen[2],Ncrit,Re, M, NoIter)
print name,getLDmax(name)








