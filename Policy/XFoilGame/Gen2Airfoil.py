from BezierN import BezierN
from testXFoil import *
import matplotlib.pyplot as plt



def gen2airfoil(gen,name):
    upx = [0, 0,  0.25,0.5,0.75,1]
    downx = upx

    upy =   [0]*6
    downy = [0]*6
              # Leading edge
    upy   [1] = gen[0]

    downy [1] = - gen[1]

              # camber + thickness
    upy   [2] = gen[2] + gen[5]
    upy   [3] = gen[3] + gen[6]
    upy   [4] = gen[4] + gen[7]
    
    downy [2] = gen[2] - gen[5]
    downy [3] = gen[3] - gen[6]
    downy [4] = gen[4] - gen[7]


    #generate foil
    n=50;
    MyBezier = BezierN(6)

    pupx  = MyBezier.interpolate(upx   ,n)
    pupy  = MyBezier.interpolate(upy   ,n)
    pdownx= MyBezier.interpolate(downx ,n)
    pdowny= MyBezier.interpolate(downy ,n)
    '''
    plt.figure()
    plt.plot(pupx, pupy, pdownx, pdowny, '-b')
    plt.xlabel('Point #')
    plt.ylabel('Max relative thickness, t/chord')
    plt.axis('equal')
    plt.grid()
    #plt.tight_layout()
    plt.show()
    '''
    # save foil
    foilfile = open(name+".dat",'w')
    foilfile.write(name+"\n")
    for i in range (n,0,-1):
         foilfile.write(  " %1.6f    %1.6f\n" %(pupx[i],pupy[i]))
    for i in range (0,n+1):
         foilfile.write(  " %1.6f    %1.6f\n" %(pdownx[i],pdowny[i]))
    foilfile.close()


'''
Re = 500
M = 0.1
NoIter = 200
Ncrit = 9.0
testgen = [ 0.05,  0.025,    0.1,  0.1,   0.05,     0.1,  0.05,  0.025   ]
#testgen = [ 0.0376,  0.0121,  0.0845, 0.1290,  0.0496,  0.0295,  0.1326,  0.0277   ]
name = "myfoil_1111"
gen2airfoil(testgen,name)
Xfoil(name,Ncrit,Re, M, NoIter)
print name,getLDmax(name)

'''