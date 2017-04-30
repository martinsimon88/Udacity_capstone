from kulfanCST import *
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess as sp
import string
from Gen2Airfoil import *
import time
import errno
from timeout import timeout
import XFoil
import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

xfoilpath = r'xfoil'
# if airfoil is defined by Bezier:
#  LEU = Leading edge up            LED = Leading edge down
#  C25 = Camber at 25%              T25 = Camber at 25%
#  C50 = Camber at 50%              T50 = Camber at 50%
#  C75 = Camber at 75%              T75 = Camber at 75%
#              LEU   LED     C25   C50    C75      T25   T50   T75
#also max and min is for testing Bezier
genmaxs = [1,1,1,1,1,1,1,1] # change back
genmins = [0.01, 0.01, 0, 0, 0, 0.01, 0.01, 0.01] #change back
ngen= len (genmaxs)

#if airfoil is CST:
#actions = [LEU, U1, U2, TEU, LED, D1, D2, TED]
#BOUND_LOW, BOUND_UP = [0.1, 0.1, 0, 0.1, -0.5, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, -0.1, 1, 1, 1]
#CST new boundaries
#[0.1, 0.1, 0, 0.1, -0.5, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, -0.1, 0.5, 0.5, 0.5]

class XFoilGame():

    def __init__(self):
        self.iteration = [0,0,0,0,0,0,0,0]
        self.Re = 300000
        self.M = [0.04, 0.1]
        self.NoIter = 200
        self.Ncrit = 9.0
        self.foilnum = 0
        self.airfoilModel = 'Bezier' #'CST' #'Bezier'

    def newGame(self, variables, type):
        if type == 'CST':
            self.airfoilModel = 'CST'
            os.chdir("/home/simonx/Documents/Udacity/Projects/capstone/PESM_NSGA2_CST_XFoil/Archive/CST/")
        else:
            self.airfoilModel = 'Bezier'
            os.chdir("/home/simonx/Documents/Udacity/Projects/capstone/PESM_NSGA2_CST_XFoil/Archive/Bezier/")
        #check archive for latest results
        self.checkArchive()
        #update
        self.iteration = variables
        self.foilnum += 1
        name = '%06d' %self.foilnum

        if self.airfoilModel=='CST':
            kCST=CST_shape()
            kCST2=kCST.initialize(name,variables)

        elif self.airfoilModel=='Bezier':
            self.check_range(variables)
            gen2airfoil(name, variables)
        else:
            print "airfoil model not selected"

        for n in range(0, len(self.M)):
            # Start the timer. Once 5 seconds are over, a SIGALRM signal is sent.
            signal.alarm(5)
            # This try/except loop ensures that
            #   you'll catch TimeoutException when it's sent.
            try:
                self.Xfoil(name, self.Ncrit, self.Re, self.M[n], self.NoIter)  # Whatever your function that might hang
            except TimeoutException:
                continue  # continue the for loop if function A takes more than 5 second
            else:
                # Reset the alarm
                signal.alarm(0)
        #with timeout(seconds=1, exception=RuntimeError):

        #self.Xfoil(name,self.Ncrit,self.Re, self.M, self.NoIter)
        #XFoil.XfoilT(name, self.Ncrit, self.Re, self.M, self.NoIter)

        self.saveXFoilPlot(name)
        self.writeArchiveBase(name, self.M, variables)
        return self.getObjectiveValues(name, self.M)

    def checkArchive(self):
        filelist = [f for f in os.listdir(".") if f.endswith(".dat")]
        if filelist:
            # check which is last foil number
            w = open("allVariables.log", 'r')
            for line in w:
                x = line
            self.foilnum = int(x[:6])
            w.close()
        else:
            self.foilnum = 0

    def check_range(self, gen):
        #Check if Bezier Curve is within boundaries
        for i in range(0,ngen):
            if(gen[i]>genmaxs[i]):
                gen[i]=genmaxs[i]
            if(gen[i]<genmins[i]):
                gen[i]=genmins[i]

    def Xfoil(self, name, Ncrit, Re, M, NoIter):


        print M

        def Cmd(cmd):
            ps.stdin.write(cmd + '\n')
        try:
            os.remove(name + '.log')
        except:
            pass
        # run xfoil
        ps = sp.Popen(xfoilpath, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE, shell=True)
        ps.stderr.close()
        # comand part

        Cmd('PLOP')
        Cmd('G')
        Cmd(' ')
        Cmd('load ' + name + '.dat')
        Cmd('GDES')
        Cmd('CADD')
        Cmd(' ')
        Cmd(' ')
        Cmd(' ')
        Cmd(' ')
        Cmd('PANE')
        if M<0.09:
            Cmd('SAVE')
            Cmd(name + 'XFoil.dat')

        Cmd('OPER')
        Cmd('ITER')
        Cmd('Vpar' + str(NoIter))
        Cmd('N ' + str(Ncrit))
        Cmd(' ')
        Cmd('visc ' + str(Re))
        Cmd('M ' + str(M))
        Cmd('PACC')
        Cmd(name + "M" + str(M) + '.log')  # output file
        Cmd(' ')  # no dump file
        Cmd('aseq 2 15 1')
        Cmd(' ')  # escape OPER
        Cmd('quit')  # exit
        resp = ps.stdout.read()
        print "resp:",resp   # console ouput for debug
        #ps.stdout.close()
        #ps.stdin.close()
        #ps.wait()
        #while (ps.returncode == None):
        #time.sleep(10)
        #    ps.terminate()


    def writeArchiveBase(self, name, M, variables):
        for n in range(0, len(M)):
            filename = name + "M" + str(M[n]) + ".log"
            f = open(filename, 'r')
            flines = f.readlines()
            archive = open('allResults.log', 'a')
            for i in range(12,len(flines)):
                archive.write(name + '\t' + str(M[n]) + " " + '\t' + str(flines[i]))
            archive.close()
        logVariables = open('allVariables.log', 'a')
        logVariables.write(name + "    " + "    ".join(map(str, variables)) + '\n')


    def getObjectiveValues(self, name, M):
        #Get max Lift versus Drag at Mach 0.04
        filename = name + "M" + str(M[0]) + ".log"
        f = open(filename, 'r')
        flines = f.readlines()
        Lmax = 0
        objectiveValues = []
        LL = 0
        DD = 1
        for i in range(12, len(flines)):
            # print flines[i]
            words = string.split(flines[i])
            L = float(words[1])
            D =  float(words[2])
            '''
            if (L/D > Lmax and L >1.8):
                Lmax = L/D
                LL = L
                DD = D
            '''
            if (L > Lmax and D <0.1):
                Lmax = L / D
                LL = L
                DD = D

        objectiveValues.append(DD)
        objectiveValues.append(-LL)

        # Get min Drag at Mach 0.1
        filename = name + "M" + str(M[1]) + ".log"
        f = open(filename, 'r')
        flines = f.readlines()
        Dmin = 1.0
        for i in range(12, len(flines)):
            # print flines[i]
            words = string.split(flines[i])
            L = float(words[1])
            D = float(words[2])
            if (D < Dmin and L >0.7):
                Dmin = D

        objectiveValues.append(Dmin)
        return objectiveValues


    def saveXFoilPlot(self, name):
        filename1 = str(name) + "XFoil.dat"
        filename2 = str(name) + ".dat"
        x_coor1 = []
        x_coor2 = []
        y_coor1 = []
        y_coor2 = []
        with open(filename1) as inf:
            for line in inf:
                parts = line.split()
                if len(parts) > 1:
                    x_coor1.append(parts[0])
                    y_coor1.append(parts[1])
        with open(filename2) as inf:
            for line in inf:
                parts = line.split()
                if len(parts) > 1:
                    x_coor2.append(parts[0])
                    y_coor2.append(parts[1])
        #print filename1, y_coor2, x_coor2, x_coor1, y_coor1
        plt.plot(x_coor1, y_coor1, 'r--', x_coor2, y_coor2, 'b--')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=-0.5, ymax=0.5)
        pylab.savefig(name+'XFoil.png')

'''
#CST new boundaries
#[0.1, 0.1, 0, 0.1, -0.5, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, -0.1, 0.5, 0.5, 0.5]
#crashed results:
#variables = [ 0.46807788, 0.86009368, 0.97505631, 0.97096496, -0.12053839, 0.10956593, 0.79783572, 0.25562385]
variables = [0.145477, 0.117227, 0.021392, 0.437003, -0.191609, 0.152333, 0.443196, 0.217610]
#variables = [0.5, 0.5, 0.5, 0.2, -0.1, 0.5, 0.5, 0.5] #test CST
#variables = [0.05,  0.025,    0.1,  0.1,   0.05,     0.1,  0.05,  0.025] #test Bezier
#variables = [0.1, 0.1, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05]

XFG = XFoilGame()
testCST = XFG.newGame(variables, 'CST')
print testCST
'''
