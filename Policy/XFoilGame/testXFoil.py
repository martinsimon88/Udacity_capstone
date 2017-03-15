
import subprocess as sp
import os
import shutil
import sys
import string
import time

xfoilpath = r'xfoil'




def Xfoil(name, Ncrit, Re, M, NoIter ):
    def Cmd(cmd):
        ps.stdin.write(cmd+'\n')
    try:
        os.remove(name+'.log')
    except :
        pass
    #    print ("no such file")
    # run xfoil
    ps = sp.Popen(xfoilpath ,stdin=sp.PIPE,stderr=sp.PIPE,stdout=sp.PIPE)
    ps.stderr.close()
    # comand part
    Cmd('load '+name+'.dat')
    Cmd('OPER')
    Cmd('ITER')
    Cmd('Vpar'+str(NoIter))
    Cmd('N '+str(Ncrit))
    Cmd(' ')
    Cmd('visc '+str(Re))
    Cmd('M '+str(M))
    Cmd('PACC')
    Cmd(name+'.log')  # output file
    Cmd(' ')          # no dump file
    Cmd('aseq 2 15 1')
    Cmd(' ')     # escape OPER
    Cmd('quit')  # exit
    resp = ps.stdout.read()
#    print "resp:",resp   # console ouput for debug
#    ps.stdout.close()
#    ps.stdin.close()
#    ps.wait()
    #while (ps.returncode() == None):
    #    time.sleep(1)
    #ps.kill()
name = "000001"

Xfoil(name,9.0,500, 0.1, 200)

def getLDmax(name):
    filename = name+".log"
    f = open(filename, 'r')
    flines = f.readlines()
    LDmax = 0
    for i in range(12,len(flines)):
        #print flines[i]
        words = string.split(flines[i])
        LD = float(words[1])/float(words[2])
        if(LD>LDmax):
            LDmax = LD
    return LDmax

#print getLDmax(name)