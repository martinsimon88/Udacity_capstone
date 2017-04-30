import os
import subprocess as sp
import errno
from timeout import timeout
import time

xfoilpath = r'xfoil'

@timeout(20, os.strerror(errno.ETIMEDOUT))
def XfoilT(name, Ncrit, Re, M, NoIter):

    for n in range(0, len(M)):
        print M[n]

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
        #Cmd('GDES')
        #Cmd('CADD')
        #Cmd(' ')
        #Cmd(' ')
        #Cmd(' ')
        #Cmd(' ')
        Cmd('PANE')
        if n<1:
            Cmd('SAVE')
            Cmd(name + 'XFoil.dat')

        Cmd('OPER')
        Cmd('ITER')
        Cmd('Vpar' + str(NoIter))
        Cmd('N ' + str(Ncrit))
        Cmd(' ')
        Cmd('visc ' + str(Re))
        Cmd('M ' + str(M[n]))
        Cmd('PACC')
        Cmd(name + "M" + str(M[n]) + '.log')  # output file
        Cmd(' ')  # no dump file
        Cmd('aseq 2 15 1')
        Cmd(' ')  # escape OPER
        Cmd('quit')  # exit
        resp = ps.stdout.read()
        print "resp:",resp   # console ouput for debug
        #ps.stdout.close()
        #ps.stdin.close()
        #ps.wait()
        while (ps.returncode == None):
            time.sleep(5)
        ps.terminate()