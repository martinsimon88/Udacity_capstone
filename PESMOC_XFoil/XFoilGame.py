from Gen2Airfoil import *
from kulfanCST import *
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess as sp
import string

'''
# remove all files from last run
filelist = [ f for f in os.listdir(".") if f.endswith(".dat") ]
for f in filelist:
    os.remove(f)
#    print f
filelist = [ f for f in os.listdir(".") if f.endswith(".log") ]
for f in filelist:
    os.remove(f)
'''

xfoilpath = r'xfoil'
# if airfoil is defined by BezierN:
#  LEU = Leading edge up            LED = Leading edge down
#  C25 = Camber at 25%              T25 = Camber at 25%
#  C50 = Camber at 50%              T50 = Camber at 50%
#  C75 = Camber at 75%              T75 = Camber at 75%
#              LEU   LED     C25   C50    C75      T25   T50   T75
#also max and min is for testing BezierN
genmaxs = [10, 10, 10, 10, 10, 10, 10, 10]
genmins = [-10, -10, -10, -10, -10, -10, -10, -10]

ngen= len (genmaxs)

class XFoilGame():

    def __init__(self):
        self.child = [0,0,0,0,0,0,0,0]
        self.step=0.05
        self.Re = 300000
        self.M1 = 0.04
        self.M2 = 0.1
        self.NoIter = 200
        self.Ncrit = 9.0
        self.state = None
        self.foilnum = 0
        self.airfoilModel = 'CST' #'CST' #'BezierN'

    def Xfoil(self, name, Ncrit, Re, M1, M2, NoIter):
        def Cmd(cmd):
            ps.stdin.write(cmd + '\n')

        try:
            os.remove(name + '.log')
        except:
            pass
        # print ("no such file")
        # run xfoil
        ps = sp.Popen(xfoilpath, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        ps.stderr.close()
        # comand part
        Cmd('load ' + name + '.dat')
        Cmd('OPER')
        Cmd('ITER')
        Cmd('Vpar' + str(NoIter))
        Cmd('N ' + str(Ncrit))
        Cmd(' ')
        Cmd('visc ' + str(Re))
        Cmd('M ' + str(M1))
        Cmd('PACC')
        Cmd(name + str(M1) + '.log')  # output file
        Cmd(' ')  # no dump file
        Cmd('aseq 2 15 0.5')
        Cmd(' ')  # escape OPER
        Cmd('quit')  # exit
        resp = ps.stdout.read()


        def Cmd(cmd):
            ps.stdin.write(cmd + '\n')

        try:
            os.remove(name + '.log')
        except:
            pass
        # print ("no such file")
        # run xfoil
        ps = sp.Popen(xfoilpath, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        ps.stderr.close()
        Cmd('load ' + name + '.dat')
        Cmd('OPER')
        Cmd('ITER')
        Cmd('Vpar' + str(NoIter))
        Cmd('N ' + str(Ncrit))
        Cmd(' ')
        Cmd('visc ' + str(Re))
        Cmd('M ' + str(M1))
        Cmd('PACC')
        Cmd(name + str(M2) + '.log')  # output file
        Cmd(' ')  # no dump file
        Cmd('aseq 2 15 0.5')
        Cmd(' ')  # escape OPER
        Cmd('quit')  # exit
        resp = ps.stdout.read()
        #    print "resp:",resp   # console ouput for debug
        #    ps.stdout.close()
        #    ps.stdin.close()
        #    ps.wait()
        # while (ps.returncode() == None):
        #    time.sleep(1)
        # ps.kill()

    def newborn(self, gen):
        self.foilnum
        self.foilnum+=1
        return [0,gen,'%06d' %self.foilnum]

    def check_range(self, gen):
        for i in range(0,ngen):
            if(gen[i]>genmaxs[i]):
                gen[i]=genmaxs[i]
            if(gen[i]<genmins[i]):
                gen[i]=genmins[i]



    def newGame(self, actions):
        self.checkArchive()
        self.child = actions

        testgen = self.newborn(self.child)
        name=testgen[2]

        if self.airfoilModel=='CST':
            kCST=CST_shape()
            kCST2=kCST.initialize(name,actions)

        elif self.airfoilModel=='BezierN':
            self.check_range(self.child)
            testgen = self.newborn(self.child)
            gen2airfoil(name, testgen[1])

        else:
            print "airfoil model not selected"

        self.Xfoil(name,self.Ncrit,self.Re, self.M1, self.M2, self.NoIter)
        self.writeArchiveBase(name, self.M1, self.M2, actions)
        return self.getObjectiveValues(name)

    def reset(self):
        self.a

        return state

    def checkArchive(self):
        filelist = [f for f in os.listdir(".") if f.endswith(".dat")]
        if filelist:
            # check which is last foil number
            w = open("allActions.log", 'r')
            for line in w:
                x = line
            self.foilnum = int(x[:6])
            w.close()
        else:
            self.foilnum = 0

    def writeArchiveBase(self, name, M1, M2, actions):
        filename = name + str(M1) + ".log"
        f = open(filename, 'r')
        flines = f.readlines()
        archive = open('allResults.log', 'a')
        for i in range(12,len(flines)):
            archive.write(filename + str(flines[i]))
        archive.close()
        logActions = open('allActions.log', 'a')
        logActions.write(filename + "    " + "    ".join(map(str, actions)))

        filename = name + str(M2) + ".log"
        f = open(filename, 'r')
        flines = f.readlines()
        archive = open('allResults.log', 'a')
        for i in range(12, len(flines)):
            archive.write(filename + str(flines[i]))
        archive.close()
        logActions = open('allActions.log', 'a')
        logActions.write(filename + "    " + "    ".join(map(str, actions)) + '\n')

    def getLDmax(self, name):
        filename = name + "0.04.log"
        f = open(filename, 'r')
        flines = f.readlines()
        LDmax = 0
        for i in range(12, len(flines)):
            # print flines[i]
            words = string.split(flines[i])
            LD = float(words[1]) / float(words[2])
            if (LD > LDmax):
                LDmax = LD
        return LDmax

    def getObjectiveValues(self, name):
        filename = name + "0.04.log"
        f = open(filename, 'r')
        flines = f.readlines()
        Lmax = 0
        Dmin = 10

        LL=0.0
        DD=0.0
        for i in range(12, len(flines)):
            # print flines[i]
            words = string.split(flines[i])
            L = float(words[1])
            D =  float(words[2])
            if (L/D > Lmax):
                Lmax = L/D
                DD=D
                LL=-L

            '''
            if (L > Lmax):
                Lmax = L
            if (D < Dmin):
                Dmin = D
            DD=Dmin
            LL=-Lmax
            '''



        return LL,DD

'''
os.chdir("/home/simonx/Documents/Udacity/Projects/capstone/PESMOC_XFoil/Archive")
LEU = 0.5 #min 0.02 max 0.5
LED = -0.5 #min -0.5 max -0.02
TEU = 0.5
TED = 0.3

'''
os.chdir("/home/simonx/Documents/Udacity/Projects/capstone/PESMOC_XFoil/Archive")
#actions = [LEU, TEU, LED, TED]
actions = [0.5, 0.5, 0.5, 0.2, -0.5, -0.5, -0.3, -0.05] #0.01 0.01 -0.1 -0.1 -0.1
XFG = XFoilGame()
testCST = XFG.newGame(actions)
print testCST

