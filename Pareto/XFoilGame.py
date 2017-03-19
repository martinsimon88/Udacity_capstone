from Gen2Airfoil import *
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
# Airfoil is defined by
#  LEU = Leading edge up            LED = Leading edge down
#  C25 = Camber at 25%              T25 = Camber at 25%
#  C50 = Camber at 50%              T50 = Camber at 50%
#  C75 = Camber at 75%              T75 = Camber at 75%
#              LEU   LED     C25   C50    C75      T25   T50   T75
genmaxs = [    0.2,  0.2,    0.2,  0.2,   0.2,     0.2,  0.2,  0.1   ]
genmins = [    0.01,  0.01,    0.01,  0.01,   0.01,     0.01,  0.01,  0.01   ]

ngen= len (genmaxs)

class XFoilGame():

    def __init__(self):
        self.child = [0,0,0,0,0,0,0,0]
        self.step=0.05
        self.Re = 500
        self.M = 0.1
        self.NoIter = 200
        self.Ncrit = 9.0
        self.state = None
        self.foilnum = 0

    def Xfoil(self, name, Ncrit, Re, M, NoIter):
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
        Cmd('M ' + str(M))
        Cmd('PACC')
        Cmd(name + '.log')  # output file
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



    def takeAction(self, actions):

        self.child=actions
        self.check_range(self.child)
        return self.newborn(self.child)



    def newGame(self, actions):
        self.checkArchive()
        testgen = self.takeAction(actions)
        name=testgen[2]
        gen2airfoil(testgen[1],name)
        self.Xfoil(testgen[2],self.Ncrit,self.Re, self.M, self.NoIter)
        self.writeArchiveBase(name, actions)
        print name,self.getLDmax(name)

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

    def writeArchiveBase(self, name, actions):
        filename = name + ".log"
        f = open(filename, 'r')
        flines = f.readlines()
        archive = open('allResults.log', 'a')
        for i in range(12,len(flines)):
            archive.write(filename + str(flines[i]))
        archive.close()
        logActions = open('allActions.log', 'a')
        logActions.write(filename + "    " + "    ".join(map(str, actions)) + '\n')

    def getLDmax(self, name):
        filename = name + ".log"
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
